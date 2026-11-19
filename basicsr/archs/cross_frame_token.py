"""跨帧 Token 交互模块：Tokenizer、元信息位置/几何偏置、质量偏置器、两层 Pre-LN Transformer 与空间回注。"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_util import trunc_normal_


class TokenizerBase(nn.Module):
    """区域特征到 token 描述子的基类。"""

    def forward(self, u_blocks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GridPatchTokenizer(TokenizerBase):
    """主模型 tokenizer：DWConv3x3(C=48) -> GELU -> AdaptiveAvgPool2d(2,4) -> LN(384) -> Linear(384,96)。

    参数预算（C=48, D=96）:
      - DWConv3x3: 48 * (3*3 + 1) = 480
      - LN(384): 2 * 384 = 768
      - Linear(384, 96): 384 * 96 + 96 = 36,960
      合计 = 38,208
    """

    def __init__(self, in_ch: int = 48, token_dim: int = 96, pool_h: int = 2, pool_w: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.pool_h = pool_h
        self.pool_w = pool_w
        flat_dim = in_ch * pool_h * pool_w

        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.norm = nn.LayerNorm(flat_dim)
        self.proj = nn.Linear(flat_dim, token_dim, bias=True)

    def forward(self, u_blocks: torch.Tensor) -> torch.Tensor:
        # u_blocks: [N, C, h_patch, w_patch]
        x = self.act(self.dwconv(u_blocks))
        x = self.pool(x)  # [N, C, pool_h, pool_w] 按通道、行、列固定顺序展开
        x = x.flatten(1)
        x = self.norm(x)
        return self.proj(x)


class StatsPatchTokenizer(TokenizerBase):
    """统计对照 tokenizer：逐通道均值与 sqrt(总体方差+1e-6) 拼接后经 LN(2C) -> Linear(2C, D)。

    参数预算（C=48, D=96）: LN(96)=192 + Linear(96,96)=9,312 = 9,504（比主 tokenizer 少 28,704）。
    """

    def __init__(self, in_ch: int = 48, token_dim: int = 96, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(in_ch * 2)
        self.proj = nn.Linear(in_ch * 2, token_dim, bias=True)

    def forward(self, u_blocks: torch.Tensor) -> torch.Tensor:
        # u_blocks: [N, C, h_patch, w_patch]
        mean = u_blocks.mean(dim=(-2, -1))
        var = u_blocks.var(dim=(-2, -1), unbiased=False)
        std = torch.sqrt(var + self.eps)
        feat = torch.cat([mean, std], dim=-1)
        return self.proj(self.norm(feat))


class QualityBiasPredictor(nn.Module):
    """单层双因子质量偏置器（指导书第 7.1 节）：仅读取加位置嵌入前的原始描述子 z_i 与 q_i。

    结构：ln_q = LN(96) + mlp_src(97->16->4, GELU) + mlp_tgt(97->16->4, GELU)。
    参数：192 + 2*(97*16+16 + 16*4+4) = 192 + 2*1636 = 3,464 / 层（两层共 6,928）。
    """

    def __init__(self, token_dim: int = 96, num_heads: int = 4, hidden_dim: int = 16, src_only: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.src_only = src_only
        self.ln_q = nn.LayerNorm(token_dim)
        self.mlp_src = nn.Sequential(
            nn.Linear(token_dim + 1, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads, bias=True),
        )
        # src_only 模式下不实例化 mlp_tgt，防止 DDP 出现未使用参数
        self.mlp_tgt = None if src_only else nn.Sequential(
            nn.Linear(token_dim + 1, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads, bias=True),
        )
        self.reset_output_zeros()

    def reset_output_zeros(self):
        """只将输出层置零，隐藏层保留正常初始化，使初始 c=0、d=2*sigmoid(0)=1。"""
        for mlp in (self.mlp_src, self.mlp_tgt):
            if mlp is None:
                continue
            out_linear = mlp[-1]
            nn.init.zeros_(out_linear.weight)
            if out_linear.bias is not None:
                nn.init.zeros_(out_linear.bias)

    def forward(self, z_raw: torch.Tensor, q_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_raw: [B, N, D] 未加位置嵌入的描述子
        q_tok: [B, N, 1] 质量来源比例
        返回:
          bias: [B, num_heads, N, N]
          d_tgt: [B, N, num_heads]
          c_src: [B, N, num_heads]
        """
        feat = torch.cat([self.ln_q(z_raw), q_tok], dim=-1)
        c_src = self.mlp_src(feat)  # [B, N, H]
        if self.src_only:
            d_tgt = torch.ones_like(c_src)
        else:
            d_tgt = 2.0 * torch.sigmoid(self.mlp_tgt(feat))  # [B, N, H]
        # b_quality^h(i,j) = d_i^h * c_j^h
        bias = torch.einsum('bih,bjh->bhij', d_tgt, c_src)
        return bias, d_tgt, c_src


class TokenTransformerLayer(nn.Module):
    """带显式相对几何偏置、质量偏置与动态掩码的 Pre-LN Transformer 层。"""

    def __init__(
        self,
        token_dim: int = 96,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        use_geom_bias: bool = True,
        use_row_bias: bool = True,
        use_ground_bias: bool = True,
        max_frame_rel: int = 8,
        max_ground_rel: int = 11,
        max_row_rel: int = 3,
        quality_mode: str = 'none',
        quality_src_only: bool = False,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(f'token_dim={token_dim} must be divisible by num_heads={num_heads}')
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(token_dim)
        self.qkv = nn.Linear(token_dim, token_dim * 3, bias=True)
        self.proj = nn.Linear(token_dim, token_dim, bias=True)

        self.norm2 = nn.LayerNorm(token_dim)
        ffn_hidden = int(round(token_dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_hidden, bias=True),
            nn.GELU(),
            nn.Linear(ffn_hidden, token_dim, bias=True),
        )

        # 相对几何偏置查找表（第 5.2 节）
        self.use_geom_bias = use_geom_bias
        self.use_row_bias = use_row_bias and use_geom_bias
        self.use_ground_bias = use_ground_bias and use_geom_bias
        self.max_frame_rel = max_frame_rel
        self.max_ground_rel = max_ground_rel
        self.max_row_rel = max_row_rel

        if self.use_geom_bias:
            self.rel_frame_table = nn.Parameter(torch.zeros(2 * max_frame_rel + 1, num_heads))
        if self.use_ground_bias:
            self.rel_ground_table = nn.Parameter(torch.zeros(2 * max_ground_rel + 1, num_heads))
        if self.use_row_bias:
            self.rel_row_table = nn.Parameter(torch.zeros(2 * max_row_rel + 1, num_heads))

        self.quality_bias = (
            QualityBiasPredictor(token_dim=token_dim, num_heads=num_heads, src_only=quality_src_only)
            if quality_mode != 'none'
            else None
        )

    def reset_zero_init_branches(self):
        if self.use_geom_bias:
            nn.init.zeros_(self.rel_frame_table)
        if self.use_ground_bias:
            nn.init.zeros_(self.rel_ground_table)
        if self.use_row_bias:
            nn.init.zeros_(self.rel_row_table)
        if self.quality_bias is not None:
            self.quality_bias.reset_output_zeros()

    def _compute_geom_bias(
        self,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        r_idx: torch.Tensor,
        valid_ground_pair: torch.Tensor | None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        t_idx, s_idx, r_idx: [B, N] long
        返回: [B, num_heads, N, N]
        """
        bsz, n_tok = t_idx.shape
        device = t_idx.device
        bias = torch.zeros((bsz, self.num_heads, n_tok, n_tok), dtype=dtype, device=device)
        if not self.use_geom_bias:
            return bias

        dt = (t_idx.unsqueeze(2) - t_idx.unsqueeze(1)).clamp(-self.max_frame_rel, self.max_frame_rel) + self.max_frame_rel
        b_t = self.rel_frame_table[dt].permute(0, 3, 1, 2).to(dtype)  # [B, H, N, N]
        bias = bias + b_t

        if self.use_ground_bias:
            ds = (s_idx.unsqueeze(2) - s_idx.unsqueeze(1)).clamp(-self.max_ground_rel, self.max_ground_rel) + self.max_ground_rel
            b_s = self.rel_ground_table[ds].permute(0, 3, 1, 2).to(dtype)
            if valid_ground_pair is not None:
                b_s = b_s * valid_ground_pair.unsqueeze(1).to(dtype)
            bias = bias + b_s

        if self.use_row_bias:
            dr = (r_idx.unsqueeze(2) - r_idx.unsqueeze(1)).clamp(-self.max_row_rel, self.max_row_rel) + self.max_row_rel
            b_r = self.rel_row_table[dr].permute(0, 3, 1, 2).to(dtype)
            if valid_ground_pair is not None:
                b_r = b_r * valid_ground_pair.unsqueeze(1).to(dtype)
            bias = bias + b_r

        return bias

    def forward(
        self,
        z: torch.Tensor,
        z_raw: torch.Tensor,
        q_tok: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        r_idx: torch.Tensor,
        attn_mask: torch.Tensor,
        valid_ground_pair: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        attn_mask: [B, 1, N, N], 允许为 0.0, 屏蔽为 -inf
        """
        bsz, n_tok, dim = z.shape
        x_norm = self.norm1(z)
        qkv = self.qkv(x_norm).reshape(bsz, n_tok, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_h]

        logits = (q @ k.transpose(-2, -1)) * self.scale
        logits = logits + self._compute_geom_bias(t_idx, s_idx, r_idx, valid_ground_pair, dtype=logits.dtype)

        q_info = {}
        if self.quality_bias is not None:
            b_qual, d_tgt, c_src = self.quality_bias(z_raw, q_tok)
            logits = logits + b_qual
            q_info = {'d_tgt': d_tgt, 'c_src': c_src, 'b_qual': b_qual}

        logits = logits + attn_mask
        attn = torch.softmax(logits, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(bsz, n_tok, dim)
        z = z + self.proj(out)
        z = z + self.ffn(self.norm2(z))
        return z, attn, q_info


class CrossFrameTokenInteraction(nn.Module):
    """九帧 SAR 跨帧 Token 交互模块（支持首轮 B/C/D1/D0 与第二轮全部消融）。"""

    GRANULARITY_GRID = {
        'region': (4, 4),
        'strip': (1, 4),
        'global': (1, 1),
    }

    def __init__(
        self,
        in_ch: int = 48,
        token_dim: int = 96,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        num_layers: int = 2,
        num_frames: int = 9,
        granularity: str = 'region',
        tokenizer_type: str = 'grid',
        mask_mode: str = 'full',
        causal_lookahead: Sequence[int] | None = None,
        use_frame_embed: bool = True,
        use_geom_bias: bool = True,
        use_ground_bias: bool = True,
        use_row_bias: bool | None = None,
        quality_mode: str = 'none',
        quality_src_only: bool = False,
        writeback_mode: str = 'film',
        use_delta_ln: bool = False,
        use_detail: bool = False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.granularity = granularity
        self.tokenizer_type = tokenizer_type
        self.mask_mode = mask_mode
        self.causal_lookahead = list(causal_lookahead) if causal_lookahead is not None else [1, 0]
        if len(self.causal_lookahead) != num_layers:
            raise ValueError(f'causal_lookahead length {len(self.causal_lookahead)} != num_layers {num_layers}')

        self.use_frame_embed = use_frame_embed
        self.use_geom_bias = use_geom_bias
        self.use_ground_bias = use_ground_bias
        self.quality_mode = quality_mode
        self.writeback_mode = writeback_mode
        self.use_delta_ln = use_delta_ln
        self.use_detail = use_detail

        if granularity not in ('region', 'strip', 'global', 'hierarchical'):
            raise ValueError(f'Unsupported granularity: {granularity}')
        if mask_mode not in ('intra', 'full', 'causal', 'same_ground'):
            raise ValueError(f'Unsupported mask_mode: {mask_mode}')
        if quality_mode not in ('none', 'real', 'constant', 'shuffle'):
            raise ValueError(f'Unsupported quality_mode: {quality_mode}')
        if writeback_mode not in ('film', 'additive'):
            raise ValueError(f'Unsupported writeback_mode: {writeback_mode}')

        # 1. Tokenizer 与位置嵌入
        if granularity == 'hierarchical':
            self.grid_h, self.grid_w = 1, 4
            self.global_tokenizer = self._make_tokenizer(tokenizer_type, in_ch, token_dim, pool_h=2, pool_w=4)
            self.strip_tokenizer = self._make_tokenizer(tokenizer_type, in_ch, token_dim, pool_h=2, pool_w=4)
            self.type_embed = nn.Embedding(2, token_dim)  # 0=global, 1=strip
            has_row_bias = False if use_row_bias is None else use_row_bias
            has_ground_bias = use_ground_bias
            max_row_rel = 0
        else:
            self.grid_h, self.grid_w = self.GRANULARITY_GRID[granularity]
            self.tokenizer = self._make_tokenizer(tokenizer_type, in_ch, token_dim, pool_h=2, pool_w=4)
            has_row_bias = (self.grid_h > 1) if use_row_bias is None else use_row_bias
            has_ground_bias = (self.grid_w > 1) and use_ground_bias
            max_row_rel = max(self.grid_h - 1, 0)

        if self.use_frame_embed:
            self.frame_embed = nn.Embedding(num_frames, token_dim)
        self.row_embed = nn.Embedding(self.grid_h, token_dim)
        self.col_embed = nn.Embedding(self.grid_w, token_dim)

        # 2. 两层序列 Transformer
        self.layers = nn.ModuleList([
            TokenTransformerLayer(
                token_dim=token_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                use_geom_bias=use_geom_bias,
                use_row_bias=has_row_bias,
                use_ground_bias=has_ground_bias,
                max_frame_rel=num_frames - 1,
                max_ground_rel=(num_frames - 1) + (self.grid_w - 1),
                max_row_rel=max_row_rel,
                quality_mode=quality_mode,
                quality_src_only=quality_src_only,
            )
            for _ in range(num_layers)
        ])

        # 3. 空间回注模块（第 8.1 节）
        if use_delta_ln:
            self.delta_ln = nn.LayerNorm(token_dim)
        else:
            self.delta_ln = None

        if granularity == 'hierarchical':
            # 将每帧 global delta 广播融合进 1x4 strip delta
            self.hier_fuse = nn.Linear(token_dim * 2, token_dim, bias=True)

        self.wb_pre = nn.Conv2d(token_dim, in_ch, kernel_size=1, bias=True)
        self.wb_dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True)
        self.wb_act = nn.GELU()
        out_ch = in_ch * 2 if writeback_mode == 'film' else in_ch
        self.wb_out = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

        # 4. 第二轮可选：同地面细节伴随传递（第 8.2 节）
        if use_detail:
            self.detail_pv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
            self.detail_pd = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)
        else:
            self.detail_pv = None
            self.detail_pd = None

        self._init_non_zero_weights()
        self.reset_zero_init_branches()

    @staticmethod
    def _make_tokenizer(tokenizer_type: str, in_ch: int, token_dim: int, pool_h: int, pool_w: int) -> TokenizerBase:
        if tokenizer_type == 'grid':
            return GridPatchTokenizer(in_ch=in_ch, token_dim=token_dim, pool_h=pool_h, pool_w=pool_w)
        if tokenizer_type == 'stats':
            return StatsPatchTokenizer(in_ch=in_ch, token_dim=token_dim)
        raise ValueError(f'Unsupported tokenizer_type: {tokenizer_type}')

    def _init_non_zero_weights(self):
        """标准初始化可学习嵌入与投影权重（零初始化分支在 reset_zero_init_branches 中处理）。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_zero_init_branches(self):
        """确保回注出口 P_out、几何偏置表、质量输出层和 detail P_D 从恒等起点开始。"""
        nn.init.zeros_(self.wb_out.weight)
        if self.wb_out.bias is not None:
            nn.init.zeros_(self.wb_out.bias)
        if self.detail_pd is not None:
            nn.init.zeros_(self.detail_pd.weight)
            if self.detail_pd.bias is not None:
                nn.init.zeros_(self.detail_pd.bias)
        for layer in self.layers:
            layer.reset_zero_init_branches()

    def _split_grid_patches(self, u: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """按 2D 空间网格切分空间特征 U: [B, T, C, H, W] -> [B, T, grid_h, grid_w, C, H/grid_h, W/grid_w]。"""
        bsz, t_count, ch, height, width = u.shape
        if height % grid_h != 0 or width % grid_w != 0:
            raise ValueError(f'Feature size ({height}, {width}) must be divisible by grid ({grid_h}, {grid_w})')
        ph = height // grid_h
        pw = width // grid_w
        # 严格保持 2D 行/列维度语义，不展平切分
        u_grid = u.view(bsz, t_count, ch, grid_h, ph, grid_w, pw)
        return u_grid.permute(0, 1, 3, 5, 2, 4, 6).contiguous()

    def _merge_grid_patches(self, u_grid: torch.Tensor) -> torch.Tensor:
        """逆操作：[B, T, grid_h, grid_w, C, ph, pw] -> [B, T, C, grid_h*ph, grid_w*pw]。"""
        bsz, t_count, grid_h, grid_w, ch, ph, pw = u_grid.shape
        return u_grid.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bsz, t_count, ch, grid_h * ph, grid_w * pw)

    def _prepare_quality_tokens(
        self,
        token_quality: torch.Tensor | None,
        bsz: int,
        t_count: int,
        n_spatial: int,
        device: torch.device,
        is_hierarchical: bool,
    ) -> torch.Tensor:
        """生成与 token 序列化对齐的 q_tok: [B, N, 1]。"""
        if self.quality_mode == 'none' or self.quality_mode == 'constant':
            n_total = t_count * (1 + n_spatial) if is_hierarchical else t_count * n_spatial
            return torch.full((bsz, n_total, 1), 0.5, dtype=torch.float32, device=device)

        if token_quality is None:
            raise ValueError(f'token_quality is required when quality_mode={self.quality_mode}')

        # token_quality: [B, T, grid_h, grid_w, 1] 或 [B, T, 4, 4, 1]
        if token_quality.shape[2] != self.grid_h or token_quality.shape[3] != self.grid_w:
            # 当 K=1 或 K=4 时从 4x4 质量标签均值投影到当前网格
            q_avg = token_quality.squeeze(-1).view(bsz, t_count, self.grid_h, -1, self.grid_w, -1).mean(dim=(3, 5))
            q_grid = q_avg.unsqueeze(-1)
        else:
            q_grid = token_quality

        q_strip_or_reg = q_grid.reshape(bsz, t_count, n_spatial, 1)
        if is_hierarchical:
            # global token 的质量为整帧平均
            q_glob = q_strip_or_reg.mean(dim=2, keepdim=True)  # [B, T, 1, 1]
            q_all = torch.cat([q_glob, q_strip_or_reg], dim=2).reshape(bsz, t_count * (1 + n_spatial), 1)
        else:
            q_all = q_strip_or_reg.reshape(bsz, t_count * n_spatial, 1)

        if self.quality_mode == 'shuffle':
            # 在同批内部沿 token 维度打乱质量标签（保留分布，破坏对应关系）
            perm = torch.randperm(q_all.shape[1], device=device)
            q_all = q_all[:, perm, :]
        return q_all

    def _build_layer_mask(
        self,
        layer_idx: int,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        r_idx: torch.Tensor,
        is_strip_mask: torch.Tensor | None,
        masked_source_frames: torch.Tensor | Sequence[int] | None,
    ) -> torch.Tensor:
        """构造第 layer_idx 层的 attention mask: [B, 1, N, N]，允许为 0.0，禁止为 -inf。"""
        bsz, n_tok = t_idx.shape
        device = t_idx.device
        ti = t_idx.unsqueeze(2)  # [B, N, 1]
        tj = t_idx.unsqueeze(1)  # [B, 1, N]

        if self.mask_mode == 'intra':
            allow = ti == tj
        elif self.mask_mode == 'full':
            allow = torch.ones((bsz, n_tok, n_tok), dtype=torch.bool, device=device)
        elif self.mask_mode == 'causal':
            lookahead = self.causal_lookahead[layer_idx]
            allow = tj <= (ti + lookahead)
        elif self.mask_mode == 'same_ground':
            same_frame = ti == tj
            same_pos = (s_idx.unsqueeze(2) == s_idx.unsqueeze(1)) & (r_idx.unsqueeze(2) == r_idx.unsqueeze(1))
            if is_strip_mask is not None:
                both_strip = is_strip_mask.unsqueeze(2) & is_strip_mask.unsqueeze(1)
                same_pos = same_pos & both_strip
            allow = same_frame | same_pos
        else:
            raise ValueError(f'Unknown mask_mode: {self.mask_mode}')

        # 运行时来源屏蔽（第 13.3 节：仅禁止作为跨帧 K/V，不屏蔽同帧自身）
        if masked_source_frames is not None:
            if not torch.is_tensor(masked_source_frames):
                masked_source_frames = torch.tensor(list(masked_source_frames), dtype=torch.long, device=device)
            else:
                masked_source_frames = masked_source_frames.to(device=device, dtype=torch.long)
            if masked_source_frames.numel() > 0:
                is_masked_src = (tj.unsqueeze(-1) == masked_source_frames.view(1, 1, 1, -1)).any(dim=-1)
                allow = allow & ((ti == tj) | (~is_masked_src))

        mask = torch.zeros((bsz, 1, n_tok, n_tok), dtype=torch.float32, device=device)
        mask.masked_fill_(~allow.unsqueeze(1), float('-inf'))
        return mask

    def _apply_detail_path(
        self,
        u_grid_before: torch.Tensor,
        u_updated: torch.Tensor,
        last_attn: torch.Tensor,
        last_mask: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        r_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """第二轮可选同地面细节伴随传递（指导书第 8.2 节）。"""
        bsz, t_count, grid_h, grid_w, ch, ph, pw = u_grid_before.shape
        n_spatial = grid_h * grid_w
        # 将更新前的空间块投影为载荷，仅读取更新前各帧特征，避免递推依赖
        u_flat_before = u_grid_before.reshape(bsz * t_count * n_spatial, ch, ph, pw)
        pv_flat = self.detail_pv(u_flat_before).reshape(bsz, t_count * n_spatial, ch, ph, pw)

        # 多头平均注意力
        a_bar = last_attn.mean(dim=1)  # [B, N, N]
        visible = torch.isfinite(last_mask[:, 0])  # [B, N, N]

        if self.granularity == 'hierarchical':
            # 1+4 模式下提取 strip-to-strip (每个帧内索引 1..4)
            strip_pos = []
            for ft in range(t_count):
                base = ft * (1 + n_spatial)
                strip_pos.extend(range(base + 1, base + 1 + n_spatial))
            idx_t = torch.tensor(strip_pos, dtype=torch.long, device=u_updated.device)
            a_bar = a_bar.index_select(1, idx_t).index_select(2, idx_t)
            visible = visible.index_select(1, idx_t).index_select(2, idx_t)
            t_sp = t_idx.index_select(1, idx_t)
            s_sp = s_idx.index_select(1, idx_t)
            r_sp = r_idx.index_select(1, idx_t)
        else:
            t_sp, s_sp, r_sp = t_idx, s_idx, r_idx

        # 严格限制：t' != t, r' == r, s' == s，且遵守最后一层 mask 与来源屏蔽
        valid_pair = (
            (t_sp.unsqueeze(2) != t_sp.unsqueeze(1))
            & (r_sp.unsqueeze(2) == r_sp.unsqueeze(1))
            & (s_sp.unsqueeze(2) == s_sp.unsqueeze(1))
            & visible
        )
        raw_w = torch.where(valid_pair, a_bar, torch.zeros_like(a_bar))
        denom = raw_w.sum(dim=-1, keepdim=True)
        has_valid = denom > 1e-6  # [B, N_sp, 1]
        safe_denom = torch.where(has_valid, denom, torch.ones_like(denom))
        beta = torch.where(has_valid, raw_w / safe_denom, torch.zeros_like(raw_w))  # [B, N_sp, N_sp]

        # 聚合对齐载荷特征: R_i = sum_j beta_{ij} P_V(U_j)
        agg = torch.einsum('bij,bjchw->bichw', beta, pv_flat)  # [B, N_sp, C, ph, pw]
        agg_flat = agg.reshape(bsz * t_count * n_spatial, ch, ph, pw)
        pd_out = self.detail_pd(agg_flat).reshape(bsz, t_count, grid_h, grid_w, ch, ph, pw)
        # 无效目标（如 s=0、s=11、全隔离 B 等）在投影后仍显式置零，排除 P_D bias 泄漏
        valid_target_mask = has_valid.reshape(bsz, t_count, grid_h, grid_w, 1, 1, 1).to(pd_out.dtype)
        pd_out = pd_out * valid_target_mask

        detail_delta = self._merge_grid_patches(pd_out)  # [B, T, C, H, W]
        info = {
            'detail_beta': beta.detach(),
            'detail_valid_counts': valid_pair.sum(dim=-1).detach(),
            'detail_delta_abs_mean': float(detail_delta.detach().abs().mean()),
        }
        return u_updated + detail_delta, info

    def forward(
        self,
        u: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        token_quality: torch.Tensor | None = None,
        masked_source_frames: torch.Tensor | Sequence[int] | None = None,
        return_telemetry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        u: [B, T, C=48, H, W]
        frame_ids: [B, T] long，默认 [0..T-1]
        token_quality: [B, T, 4, 4, 1] 或 [B, T, grid_h, grid_w, 1]
        """
        bsz, t_count, ch, height, width = u.shape
        device = u.device
        if frame_ids is None:
            frame_ids = torch.arange(t_count, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, t_count)
        else:
            frame_ids = frame_ids.to(device=device, dtype=torch.long)

        is_hier = self.granularity == 'hierarchical'
        u_grid = self._split_grid_patches(u, self.grid_h, self.grid_w)  # [B, T, gh, gw, C, ph, pw]
        n_spatial = self.grid_h * self.grid_w

        # 构造区域描述子（尚未加任何位置嵌入）
        if is_hier:
            u_full_flat = u.reshape(bsz * t_count, ch, height, width)
            z_glob = self.global_tokenizer(u_full_flat).reshape(bsz, t_count, 1, self.token_dim)
            u_strip_flat = u_grid.reshape(bsz * t_count * n_spatial, ch, u_grid.shape[-2], u_grid.shape[-1])
            z_strip = self.strip_tokenizer(u_strip_flat).reshape(bsz, t_count, n_spatial, self.token_dim)
            z_raw = torch.cat([z_glob, z_strip], dim=2).reshape(bsz, t_count * (1 + n_spatial), self.token_dim)
        else:
            u_patch_flat = u_grid.reshape(bsz * t_count * n_spatial, ch, u_grid.shape[-2], u_grid.shape[-1])
            z_raw = self.tokenizer(u_patch_flat).reshape(bsz, t_count * n_spatial, self.token_dim)

        # 元数据索引（严格由 frame_ids 与网格行列计算，支持帧排列等变性）
        r_vals = torch.arange(self.grid_h, dtype=torch.long, device=device)
        k_vals = torch.arange(self.grid_w, dtype=torch.long, device=device)
        rr, kk = torch.meshgrid(r_vals, k_vals, indexing='ij')  # [gh, gw]
        rr_flat = rr.reshape(1, 1, n_spatial).expand(bsz, t_count, n_spatial)
        kk_flat = kk.reshape(1, 1, n_spatial).expand(bsz, t_count, n_spatial)
        t_flat = frame_ids.unsqueeze(-1).expand(bsz, t_count, n_spatial)
        s_flat = t_flat + kk_flat

        if is_hier:
            t_glob = frame_ids.unsqueeze(-1)  # [B, T, 1]
            zero_rk = torch.zeros_like(t_glob)
            t_idx = torch.cat([t_glob, t_flat], dim=2).reshape(bsz, -1)
            r_idx = torch.cat([zero_rk, rr_flat], dim=2).reshape(bsz, -1)
            k_idx = torch.cat([zero_rk, kk_flat], dim=2).reshape(bsz, -1)
            s_idx = torch.cat([t_glob, s_flat], dim=2).reshape(bsz, -1)
            is_strip_mask = torch.cat(
                [torch.zeros_like(t_glob, dtype=torch.bool), torch.ones_like(t_flat, dtype=torch.bool)], dim=2
            ).reshape(bsz, -1)
            valid_ground_pair = (is_strip_mask.unsqueeze(2) & is_strip_mask.unsqueeze(1)).to(torch.float32)

            pe_glob = self.type_embed(torch.zeros_like(t_glob))
            pe_strip = self.type_embed(torch.ones_like(t_flat)) + self.row_embed(rr_flat) + self.col_embed(kk_flat)
            pos_emb = torch.cat([pe_glob, pe_strip], dim=2).reshape(bsz, -1, self.token_dim)
            if self.use_frame_embed:
                pos_emb = pos_emb + self.frame_embed(t_idx)
        else:
            t_idx = t_flat.reshape(bsz, -1)
            r_idx = rr_flat.reshape(bsz, -1)
            k_idx = kk_flat.reshape(bsz, -1)
            s_idx = s_flat.reshape(bsz, -1)
            is_strip_mask = None
            valid_ground_pair = None
            pos_emb = self.row_embed(r_idx) + self.col_embed(k_idx)
            if self.use_frame_embed:
                pos_emb = pos_emb + self.frame_embed(t_idx)

        z0 = z_raw + pos_emb
        q_tok = self._prepare_quality_tokens(token_quality, bsz, t_count, n_spatial, device, is_hier)

        z = z0
        attn_maps = []
        quality_infos = []
        last_mask = None
        for layer_idx, layer in enumerate(self.layers):
            mask = self._build_layer_mask(layer_idx, t_idx, s_idx, r_idx, is_strip_mask, masked_source_frames)
            last_mask = mask
            z, attn, q_info = layer(
                z=z,
                z_raw=z_raw,
                q_tok=q_tok,
                t_idx=t_idx,
                s_idx=s_idx,
                r_idx=r_idx,
                attn_mask=mask,
                valid_ground_pair=valid_ground_pair,
            )
            if return_telemetry:
                attn_maps.append(attn.detach())
                if q_info:
                    quality_infos.append({kk_q: vv_q.detach() for kk_q, vv_q in q_info.items()})
            else:
                attn_maps.append(attn)

        delta_z = z - z0
        if self.delta_ln is not None:
            delta_z = self.delta_ln(delta_z)

        if is_hier:
            dz_all = delta_z.reshape(bsz, t_count, 1 + n_spatial, self.token_dim)
            dz_glob = dz_all[:, :, :1, :].expand(bsz, t_count, n_spatial, self.token_dim)
            dz_strip = dz_all[:, :, 1:, :]
            dz_grid = self.hier_fuse(torch.cat([dz_glob, dz_strip], dim=-1)).reshape(
                bsz, t_count, self.grid_h, self.grid_w, self.token_dim
            )
        else:
            dz_grid = delta_z.reshape(bsz, t_count, self.grid_h, self.grid_w, self.token_dim)

        # 还原网格顺序 -> [B*T, D, gh, gw] -> 投影 96->48 -> 双线性插值 HxW -> DWConv3x3 -> GELU -> P_out
        dz_2d = dz_grid.permute(0, 1, 4, 2, 3).reshape(bsz * t_count, self.token_dim, self.grid_h, self.grid_w)
        v_low = self.wb_pre(dz_2d)
        v_up = F.interpolate(v_low, size=(height, width), mode='bilinear', align_corners=False)
        wb = self.wb_out(self.wb_act(self.wb_dwconv(v_up)))

        if self.writeback_mode == 'film':
            a_t, b_t = wb.chunk(2, dim=1)
            a_t = a_t.reshape(bsz, t_count, ch, height, width)
            b_t = b_t.reshape(bsz, t_count, ch, height, width)
            scale_gate = torch.tanh(a_t)
            u_updated = u * (1.0 + scale_gate) + b_t
        else:
            b_t = wb.reshape(bsz, t_count, ch, height, width)
            scale_gate = None
            u_updated = u + b_t

        detail_info = {}
        if self.use_detail:
            u_updated, detail_info = self._apply_detail_path(
                u_grid_before=u_grid,
                u_updated=u_updated,
                last_attn=attn_maps[-1],
                last_mask=last_mask,
                t_idx=t_idx,
                s_idx=s_idx,
                r_idx=r_idx,
            )

        if return_telemetry:
            telemetry = {
                'z_raw': z_raw.detach(),
                'delta_z_rms': float(delta_z.detach().pow(2).mean().sqrt()),
                'wb_b_rms': float(b_t.detach().pow(2).mean().sqrt()),
                'attn_layers': attn_maps,
                'quality_layers': quality_infos,
                'detail_info': detail_info,
            }
            if scale_gate is not None:
                abs_tanh = scale_gate.detach().abs()
                telemetry['tanh_abs_max'] = float(abs_tanh.max())
                telemetry['tanh_sat_gt095_ratio'] = float((abs_tanh > 0.95).float().mean())
            return u_updated, telemetry

        return u_updated
