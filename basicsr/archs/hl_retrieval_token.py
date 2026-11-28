"""UCAN-HLR 第三阶段跨帧恢复核心模块：尺度无关频域-空间联合校准（Stage1）与物理高频检索补偿（Stage2）。"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.data.sar_sequence_dataset import DEFAULT_H_INTERVALS
from .arch_util import trunc_normal_


def _compute_strip_quality_from_ids(
    frame_ids: torch.Tensor,
    grid_w: int = 4,
    frame_width: int = 512,
    stride: int = 128,
    h_intervals: tuple[tuple[int, int], ...] = DEFAULT_H_INTERVALS,
) -> torch.Tensor:
    """根据帧编号实时向量化计算每帧横向 grid_w 个条带的 H 区覆盖比例 q ∈ [0, 1]。

    返回形状: [B, T, grid_w, 1], float32
    """
    device = frame_ids.device
    patch_w = float(frame_width) / float(grid_w)
    col_starts = torch.arange(grid_w, device=device, dtype=torch.float32) * patch_w
    g0 = frame_ids.to(torch.float32).unsqueeze(-1) * float(stride) + col_starts.view(1, 1, grid_w)
    g1 = g0 + patch_w

    covered = torch.zeros_like(g0)
    for h0, h1 in h_intervals:
        overlap = (torch.clamp(g1, max=float(h1)) - torch.clamp(g0, min=float(h0))).clamp(min=0.0)
        covered = covered + overlap
    return (covered / patch_w).clamp(0.0, 1.0).unsqueeze(-1)


def _compute_pixel_quality_from_ids(
    frame_ids: torch.Tensor,
    width: int,
    frame_width: int = 512,
    stride: int = 128,
    h_intervals: tuple[tuple[int, int], ...] = DEFAULT_H_INTERVALS,
) -> torch.Tensor:
    """根据帧编号计算逐列物理 H 区覆盖质量 q_pix ∈ [0, 1]，兼容任意测试宽度与裁剪高度。

    返回形状: [B, T, 1, 1, W], float32
    """
    device = frame_ids.device
    col_scale = float(frame_width) / float(width)
    x_idx = torch.arange(width, device=device, dtype=torch.float32)
    g0 = frame_ids.to(torch.float32).unsqueeze(-1) * float(stride) + x_idx.view(1, 1, width) * col_scale
    g1 = g0 + col_scale

    covered = torch.zeros_like(g0)
    for h0, h1 in h_intervals:
        overlap = (torch.clamp(g1, max=float(h1)) - torch.clamp(g0, min=float(h0))).clamp(min=0.0)
        covered = covered + overlap
    return (covered / col_scale).clamp(0.0, 1.0).unsqueeze(2).unsqueeze(2)


def _extract_strip_quality(
    frame_ids: torch.Tensor,
    token_quality: torch.Tensor | None,
    grid_w: int = 4,
) -> torch.Tensor:
    """统一提取 [B, T, grid_w, 1] 条带质量，缺失时依据 frame_ids 物理坐标生成。"""
    if token_quality is None:
        return _compute_strip_quality_from_ids(frame_ids, grid_w=grid_w)

    q = token_quality.to(device=frame_ids.device, dtype=torch.float32)
    if q.dim() == 5:
        # [B, T, gh, gw, 1] -> 沿高度平均为条带级质量 [B, T, gw, 1]
        q = q.mean(dim=2)
    if q.shape[2] != grid_w:
        q = F.interpolate(
            q.permute(0, 1, 3, 2).reshape(-1, 1, q.shape[2]),
            size=grid_w,
            mode='linear',
            align_corners=False,
        ).reshape(q.shape[0], q.shape[1], 1, grid_w).permute(0, 1, 3, 2)
    return q.clamp(0.0, 1.0)


def _masked_frame_flags(
    frame_ids: torch.Tensor,
    masked_source_frames: torch.Tensor | Sequence[int] | None,
) -> torch.Tensor | None:
    """将 masked_source_frames 转为与 frame_ids 同形状的布尔掩膜 [B, T]。"""
    if masked_source_frames is None:
        return None
    if not torch.is_tensor(masked_source_frames):
        masked_t = torch.tensor(list(masked_source_frames), dtype=torch.long, device=frame_ids.device)
    else:
        masked_t = masked_source_frames.to(device=frame_ids.device, dtype=torch.long)
    if masked_t.numel() == 0:
        return None
    return (frame_ids.unsqueeze(-1) == masked_t.view(1, 1, -1)).any(dim=-1)


class RSDTokenizer(nn.Module):
    """高度尺度无关区域-条带描述子编码器（Region-Strip Descriptor）。

    提取每条带的通道均值 (48)、对数标准差 (48) 与横向 4 频带对数频谱能量 (192)，
    在 FP32 中计算 rfft 与二阶统计后拼接为 288 维特征并投影到 token_dim=128。
    """

    def __init__(self, in_ch: int = 48, token_dim: int = 128, grid_w: int = 4, num_freq_bands: int = 4, eps: float = 1e-6):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.grid_w = grid_w
        self.num_freq_bands = num_freq_bands
        self.eps = eps

        raw_dim = in_ch * (2 + num_freq_bands)  # 48 + 48 + 192 = 288
        self.norm = nn.LayerNorm(raw_dim)
        self.proj = nn.Linear(raw_dim, token_dim, bias=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [B, T, C, H, W] -> 返回 [B, T, grid_w, token_dim]。"""
        bsz, t_count, ch, height, width = u.shape
        if width % self.grid_w != 0:
            raise ValueError(f'Width {width} must be divisible by grid_w={self.grid_w}')
        w_strip = width // self.grid_w

        # [B, T, gw, C, H, w_strip]
        strips = u.reshape(bsz, t_count, ch, height, self.grid_w, w_strip).permute(0, 1, 4, 2, 3, 5).contiguous()
        strips_f = strips.float()

        mu = strips_f.mean(dim=(-2, -1))
        var = strips_f.var(dim=(-2, -1), unbiased=False)
        log_std = 0.5 * torch.log(var + self.eps)

        # 沿条带宽度执行 1D rfft，忽略 DC 分量后自适应划分 4 个频带
        spec = torch.fft.rfft(strips_f, dim=-1, norm='ortho')
        spec_ac = spec[..., 1:] if spec.shape[-1] > 1 else spec
        power_1d = (spec_ac.real.square() + spec_ac.imag.square()).mean(dim=-2)  # [B, T, gw, C, F_ac]
        flat_power = power_1d.reshape(-1, 1, power_1d.shape[-1])
        bands = F.adaptive_avg_pool1d(flat_power, self.num_freq_bands).reshape(
            bsz, t_count, self.grid_w, ch * self.num_freq_bands
        )
        log_bands = 0.5 * torch.log(bands + self.eps)

        feat = torch.cat([mu, log_std, log_bands], dim=-1).to(dtype=u.dtype)
        return self.proj(self.norm(feat))


class GDCFABlock(nn.Module):
    """几何感知差分跨帧注意力层（Geometry-aware Differential Cross-Frame Attention）。

    双子头 (Q1/K1, Q2/K2) 16 维差分注意力配共享零初始化帧距与地面距偏置表，
    消除跨帧共同量化噪声，并结合头级质量门控调节写回强度。
    """

    def __init__(
        self,
        token_dim: int = 128,
        num_heads: int = 4,
        ffn_ratio: float = 2.0,
        max_frame_rel: int = 8,
        max_ground_rel: int = 11,
        lambda_init: float = 0.5,
    ):
        super().__init__()
        if token_dim % (num_heads * 2) != 0:
            raise ValueError(f'token_dim={token_dim} must be divisible by 2*num_heads={2 * num_heads}')
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads       # 32
        self.sub_dim = self.head_dim // 2            # 16
        self.scale = self.sub_dim ** -0.5
        self.max_frame_rel = max_frame_rel
        self.max_ground_rel = max_ground_rel
        self.lambda_init = float(lambda_init)

        self.ln1 = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim, bias=True)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=True)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=True)
        self.o_proj = nn.Linear(token_dim, token_dim, bias=True)

        # 共享几何相对偏置表（零初始化）
        self.rel_frame_table = nn.Parameter(torch.zeros(2 * max_frame_rel + 1, num_heads))
        self.rel_ground_table = nn.Parameter(torch.zeros(2 * max_ground_rel + 1, num_heads))

        # Differential Transformer 可学习 lambda 参数化：非零微扰初始化打破 q*k=0 双线性鞍点
        self.lambda_q1 = nn.Parameter(torch.empty(num_heads, self.sub_dim))
        self.lambda_k1 = nn.Parameter(torch.empty(num_heads, self.sub_dim))
        self.lambda_q2 = nn.Parameter(torch.empty(num_heads, self.sub_dim))
        self.lambda_k2 = nn.Parameter(torch.empty(num_heads, self.sub_dim))
        for p in (self.lambda_q1, self.lambda_k1, self.lambda_q2, self.lambda_k2):
            nn.init.normal_(p, mean=0.0, std=0.1)
        self.head_norm = nn.LayerNorm(self.head_dim, elementwise_affine=True)

        # 头级质量 Sigmoid 门控：读取归一化 token 特征与标量质量 q
        self.q_gate = nn.Sequential(
            nn.Linear(token_dim + 1, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, num_heads, bias=True),
        )

        self.ln2 = nn.LayerNorm(token_dim)
        ffn_hidden = int(round(token_dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_hidden, bias=True),
            nn.GELU(),
            nn.Linear(ffn_hidden, token_dim, bias=True),
        )

    def reset_zero_init_branches(self):
        nn.init.zeros_(self.rel_frame_table)
        nn.init.zeros_(self.rel_ground_table)

    def forward(
        self,
        z: torch.Tensor,
        q_tok: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        masked_src_tok: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z: [B, N, D]
        q_tok: [B, N, 1]
        t_idx, s_idx: [B, N] long
        masked_src_tok: [B, N] bool，表示该 token 是否位于被屏蔽源帧内
        """
        bsz, n_tok, _ = z.shape
        x_norm = self.ln1(z)

        q_pair = self.q_proj(x_norm).reshape(bsz, n_tok, self.num_heads, 2, self.sub_dim)
        k_pair = self.k_proj(x_norm).reshape(bsz, n_tok, self.num_heads, 2, self.sub_dim)
        q1, q2 = q_pair[:, :, :, 0].permute(0, 2, 1, 3), q_pair[:, :, :, 1].permute(0, 2, 1, 3)
        k1, k2 = k_pair[:, :, :, 0].permute(0, 2, 1, 3), k_pair[:, :, :, 1].permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(bsz, n_tok, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 几何相对偏置 [B, H, N, N]
        dt = (t_idx.unsqueeze(2) - t_idx.unsqueeze(1)).clamp(-self.max_frame_rel, self.max_frame_rel) + self.max_frame_rel
        ds = (s_idx.unsqueeze(2) - s_idx.unsqueeze(1)).clamp(-self.max_ground_rel, self.max_ground_rel) + self.max_ground_rel
        geom_bias = (self.rel_frame_table[dt] + self.rel_ground_table[ds]).permute(0, 3, 1, 2).float()

        logits1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1)) * self.scale + geom_bias
        logits2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1)) * self.scale + geom_bias

        if masked_src_tok is not None and masked_src_tok.any():
            # 禁止跨帧读取被屏蔽来源帧，但始终保留对自身 token 的注意力通路以防全 -inf
            eye = torch.eye(n_tok, dtype=torch.bool, device=z.device).unsqueeze(0)
            block = masked_src_tok.unsqueeze(1) & (~eye)
            logits1 = logits1.masked_fill(block.unsqueeze(1), float('-inf'))
            logits2 = logits2.masked_fill(block.unsqueeze(1), float('-inf'))

        a1 = torch.softmax(logits1, dim=-1)
        a2 = torch.softmax(logits2, dim=-1)

        lam1 = torch.exp((self.lambda_q1.float() * self.lambda_k1.float()).sum(dim=-1))
        lam2 = torch.exp((self.lambda_q2.float() * self.lambda_k2.float()).sum(dim=-1))
        lam = (lam1 - lam2 + self.lambda_init).view(1, self.num_heads, 1, 1)

        attn_diff = (a1 - lam * a2).to(dtype=v.dtype)
        head_out = torch.matmul(attn_diff, v).permute(0, 2, 1, 3)  # [B, N, H, head_dim]
        head_out = self.head_norm(head_out) * (1.0 - self.lambda_init)

        gate = torch.sigmoid(self.q_gate(torch.cat([x_norm, q_tok.to(x_norm.dtype)], dim=-1))).unsqueeze(-1)
        attn_out = self.o_proj((head_out * gate).reshape(bsz, n_tok, self.token_dim))

        z = z + attn_out
        z = z + self.ffn(self.ln2(z))
        return z


class FraIRLiteFilter(nn.Module):
    """基于 8 径向频带残差响应与 FiLM 空间调制的高低区联合校准写回器（严格恒等零初始化）。"""

    def __init__(self, in_ch: int = 48, token_dim: int = 128, grid_w: int = 4, num_bands: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.grid_w = grid_w
        self.num_bands = num_bands

        self.delta_ln = nn.LayerNorm(token_dim)
        # 频域残差分支：汇聚 4 个条带上下文后预测 48 通道 × 8 径向频带的增益偏移
        self.freq_mlp = nn.Sequential(
            nn.Linear(token_dim + 1, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, in_ch * num_bands, bias=True),
        )

        # 空间 FiLM 分支：双线性上采样后经 DWConv3x3 与 zero-init 1x1 输出
        self.wb_pre = nn.Conv2d(token_dim, in_ch, kernel_size=1, bias=True)
        self.wb_dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True)
        self.wb_act = nn.GELU()
        self.wb_out = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, bias=True)
        self._band_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def reset_zero_init_branches(self):
        nn.init.zeros_(self.freq_mlp[-1].weight)
        nn.init.zeros_(self.freq_mlp[-1].bias)
        nn.init.zeros_(self.wb_out.weight)
        nn.init.zeros_(self.wb_out.bias)

    def _get_radial_band_basis(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """预计算 [num_bands, H, W//2+1] 软线性径向频带插值基底。"""
        key = (height, width, str(device))
        cached = self._band_cache.get(key)
        if cached is not None:
            return cached

        fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=torch.float32) * 2.0
        fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=torch.float32) * 2.0
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / (2.0 ** 0.5)
        pos = radius.clamp(0.0, 1.0) * float(self.num_bands - 1)
        idx_low = torch.floor(pos).long().clamp(0, self.num_bands - 2)
        alpha = pos - idx_low.to(torch.float32)

        basis = torch.zeros((self.num_bands, height, fx.shape[0]), device=device, dtype=torch.float32)
        basis.scatter_add_(0, idx_low.unsqueeze(0), (1.0 - alpha).unsqueeze(0))
        basis.scatter_add_(0, (idx_low + 1).unsqueeze(0), alpha.unsqueeze(0))
        if len(self._band_cache) < 16:
            self._band_cache[key] = basis
        return basis

    def forward(
        self,
        u: torch.Tensor,
        delta_z: torch.Tensor,
        q_strip: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """u: [B, T, C, H, W], delta_z: [B, T, gw, D], q_strip: [B, T, gw, 1]。"""
        bsz, t_count, ch, height, width = u.shape
        dz = self.delta_ln(delta_z)

        # 1. FP32 径向 8 频带残差滤波（乘子为 1 + gain_offset，严格避免零初始化时带来任意浮点误差）
        dz_mean = dz.mean(dim=2)  # [B, T, D]
        q_mean = q_strip.mean(dim=2).to(dz.dtype)  # [B, T, 1]
        gain_raw = self.freq_mlp(torch.cat([dz_mean, q_mean], dim=-1))
        gain_offset = 0.5 * torch.tanh(gain_raw.float()).reshape(bsz * t_count, ch, self.num_bands)

        basis = self._get_radial_band_basis(height, width, u.device)
        freq_delta_mask = torch.einsum('bck,khw->bchw', gain_offset, basis)

        u_flat_f = u.reshape(bsz * t_count, ch, height, width).float()
        spec = torch.fft.rfft2(u_flat_f, norm='ortho')
        delta_freq = torch.fft.irfft2(spec * freq_delta_mask, s=(height, width), norm='ortho')
        u_freq = (u_flat_f + delta_freq).to(dtype=u.dtype)

        # 2. 空间条带 FiLM 调制
        dz_2d = dz.permute(0, 1, 3, 2).reshape(bsz * t_count, self.token_dim, 1, self.grid_w)
        v_low = self.wb_pre(dz_2d)
        v_up = F.interpolate(v_low, size=(height, width), mode='bilinear', align_corners=False)
        wb = self.wb_out(self.wb_act(self.wb_dwconv(v_up)))
        a_t, b_t = wb.chunk(2, dim=1)
        scale_gate = torch.tanh(a_t)
        u_out = (u_freq * (1.0 + scale_gate) + b_t).reshape(bsz, t_count, ch, height, width)

        info = {
            'stage1_delta_freq_rms': float(delta_freq.detach().square().mean().sqrt()),
            'stage1_wb_b_rms': float(b_t.detach().float().square().mean().sqrt()),
            'stage1_tanh_abs_max': float(scale_gate.detach().abs().max()),
        }
        return u_out, info


class MultiViewStripConsensus(nn.Module):
    """同地面位置 s=0..11 的跨视角质量加权共识校正，以 zero-init Conv1x1 残差输出。"""

    def __init__(self, in_ch: int = 48, grid_w: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.grid_w = grid_w
        self.score_mlp = nn.Sequential(
            nn.Linear(in_ch + 1, 16, bias=True),
            nn.GELU(),
            nn.Linear(16, 1, bias=True),
        )
        # 初始偏置强偏向高质量来源视角
        self.quality_logit_scale = nn.Parameter(torch.tensor(4.0))
        self.cons_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)

    def reset_zero_init_branches(self):
        nn.init.zeros_(self.cons_proj.weight)
        nn.init.zeros_(self.cons_proj.bias)

    def forward(
        self,
        u: torch.Tensor,
        t_idx: torch.Tensor,
        s_idx: torch.Tensor,
        q_tok: torch.Tensor,
        masked_src_tok: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """
        u: [B, T, C, H, W]
        t_idx, s_idx: [B, T*gw] long
        q_tok: [B, T*gw, 1]
        """
        bsz, t_count, ch, height, width = u.shape
        w_strip = width // self.grid_w
        n_tok = t_count * self.grid_w

        # [B, T*gw, C, H, w_strip]
        strips = u.reshape(bsz, t_count, ch, height, self.grid_w, w_strip).permute(0, 1, 4, 2, 3, 5).reshape(
            bsz, n_tok, ch, height, w_strip
        )

        # 基于条带均值的质量自适应置信度分数
        strip_desc = strips.mean(dim=(-2, -1))  # [B, N, C]
        q_f = q_tok.to(strip_desc.dtype)
        rel_score = self.score_mlp(torch.cat([strip_desc, q_f], dim=-1)).squeeze(-1).float()
        rel_score = rel_score + self.quality_logit_scale.float() * q_tok.squeeze(-1).float()

        # 仅在物理地面位置 s 相等的多帧条带间聚合，且遵守来源掩膜
        same_ground = s_idx.unsqueeze(2) == s_idx.unsqueeze(1)  # [B, N, N]
        if masked_src_tok is not None and masked_src_tok.any():
            same_frame = t_idx.unsqueeze(2) == t_idx.unsqueeze(1)
            allow = same_ground & (same_frame | (~masked_src_tok.unsqueeze(1)))
        else:
            allow = same_ground

        logits = rel_score.unsqueeze(1).expand(bsz, n_tok, n_tok).masked_fill(~allow, float('-inf'))
        weights = torch.softmax(logits, dim=-1).to(dtype=strips.dtype)  # [B, N, N]

        strips_flat = strips.reshape(bsz, n_tok, ch * height * w_strip)
        agg_flat = torch.bmm(weights, strips_flat)
        diff_flat = (agg_flat - strips_flat).reshape(bsz * n_tok, ch, height, w_strip)

        delta_strips = self.cons_proj(diff_flat).reshape(bsz, t_count, self.grid_w, ch, height, w_strip)
        delta_u = delta_strips.permute(0, 1, 3, 4, 2, 5).reshape(bsz, t_count, ch, height, width)
        cons_rms = float(delta_u.detach().float().square().mean().sqrt())
        return u + delta_u, cons_rms


class Stage1CalibrationModule(nn.Module):
    """RG2 后第一阶段模块：RSD 编码 -> 3 层 GD-CFA -> FraIR-Lite + FiLM -> 多视角条带共识。"""

    def __init__(
        self,
        in_ch: int = 48,
        token_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_ratio: float = 2.0,
        num_frames: int = 9,
        grid_w: int = 4,
        num_freq_bands: int = 8,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.grid_w = grid_w
        self.num_frames = num_frames

        self.tokenizer = RSDTokenizer(in_ch=in_ch, token_dim=token_dim, grid_w=grid_w, num_freq_bands=4)
        self.frame_embed = nn.Embedding(num_frames, token_dim)
        self.col_embed = nn.Embedding(grid_w, token_dim)

        self.layers = nn.ModuleList([
            GDCFABlock(
                token_dim=token_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                max_frame_rel=num_frames - 1,
                max_ground_rel=(num_frames - 1) + (grid_w - 1),
                lambda_init=0.5,
            )
            for _ in range(num_layers)
        ])
        self.writeback = FraIRLiteFilter(in_ch=in_ch, token_dim=token_dim, grid_w=grid_w, num_bands=num_freq_bands)
        self.consensus = MultiViewStripConsensus(in_ch=in_ch, grid_w=grid_w)

        self._init_non_zero_weights()
        self.reset_zero_init_branches()

    def _init_non_zero_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_zero_init_branches(self):
        """确保所有残差/偏置出口严格处于恒等映射起点。"""
        for layer in self.layers:
            layer.reset_zero_init_branches()
        self.writeback.reset_zero_init_branches()
        self.consensus.reset_zero_init_branches()

    def forward(
        self,
        u: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        token_quality: torch.Tensor | None = None,
        masked_source_frames: torch.Tensor | Sequence[int] | None = None,
        return_telemetry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        bsz, t_count, _, _, _ = u.shape
        device = u.device
        if frame_ids is None:
            frame_ids = torch.arange(t_count, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, t_count)
        else:
            frame_ids = frame_ids.to(device=device, dtype=torch.long)

        q_strip = _extract_strip_quality(frame_ids, token_quality, grid_w=self.grid_w)
        z_raw = self.tokenizer(u)  # [B, T, gw, D]

        col_ids = torch.arange(self.grid_w, dtype=torch.long, device=device).view(1, 1, self.grid_w).expand(bsz, t_count, self.grid_w)
        frame_grid = frame_ids.unsqueeze(-1).expand(bsz, t_count, self.grid_w)
        s_grid = frame_grid + col_ids

        pos_emb = self.frame_embed(frame_grid) + self.col_embed(col_ids)
        z0 = (z_raw + pos_emb).reshape(bsz, t_count * self.grid_w, self.token_dim)

        t_idx = frame_grid.reshape(bsz, t_count * self.grid_w)
        s_idx = s_grid.reshape(bsz, t_count * self.grid_w)
        q_tok = q_strip.reshape(bsz, t_count * self.grid_w, 1)

        masked_frame_bool = _masked_frame_flags(frame_ids, masked_source_frames)
        masked_src_tok = None
        if masked_frame_bool is not None:
            masked_src_tok = masked_frame_bool.unsqueeze(-1).expand(bsz, t_count, self.grid_w).reshape(bsz, t_count * self.grid_w)

        z = z0
        for layer in self.layers:
            z = layer(z, q_tok=q_tok, t_idx=t_idx, s_idx=s_idx, masked_src_tok=masked_src_tok)

        delta_z = (z - z0).reshape(bsz, t_count, self.grid_w, self.token_dim)
        u_cal, tele = self.writeback(u, delta_z, q_strip)
        u_out, cons_rms = self.consensus(u_cal, t_idx=t_idx, s_idx=s_idx, q_tok=q_tok, masked_src_tok=masked_src_tok)

        if return_telemetry:
            tele['stage1_cons_delta_rms'] = cons_rms
            return u_out, tele
        return u_out


class _CrossAttnFFNBlock(nn.Module):
    """基于 PyTorch SDPA 的双源交叉注意力 + Pre-LN FFN 通用构件。"""

    def __init__(self, dim: int = 128, num_heads: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

        self.ln_ffn = nn.LayerNorm(dim)
        hidden = int(round(dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        kv_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        q_in: [N_batch, Lq, D]
        kv_in: [N_batch, Lk, D]
        kv_valid_mask: [N_batch, Lk] bool，True 表示对应 key 可参与注意力
        """
        nb, lq, _ = q_in.shape
        lk = kv_in.shape[1]

        qn = self.ln_q(q_in)
        kvn = self.ln_kv(kv_in)
        q = self.q_proj(qn).reshape(nb, lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kvn).reshape(nb, lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kvn).reshape(nb, lk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_mask = None
        if kv_valid_mask is not None and not kv_valid_mask.all():
            # 防止整行全 False 引发 SDPA softmax 产生 NaN，对全空行兜底放开第 0 槽（调用方再按 valid 标志掩去输出）
            safe_mask = kv_valid_mask.clone()
            empty_rows = ~safe_mask.any(dim=-1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            attn_mask = safe_mask[:, None, None, :].expand(nb, self.num_heads, lq, lk)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(nb, lq, self.dim)
        q_out = q_in + self.o_proj(out)
        return q_out + self.ffn(self.ln_ffn(q_out))


class _ProtoDistillBlock(nn.Module):
    """全局 H 字典蒸馏块：原型交叉汇总全图 H token + 原型间自注意力 + FFN。"""

    def __init__(self, dim: int = 128, num_heads: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 交叉注意力（使用与 _CrossAttnFFNBlock 一致的注意力机制后接自注意力与 FFN）
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cq = nn.Linear(dim, dim, bias=True)
        self.ck = nn.Linear(dim, dim, bias=True)
        self.cv = nn.Linear(dim, dim, bias=True)
        self.co = nn.Linear(dim, dim, bias=True)

        self.ln_sa = nn.LayerNorm(dim)
        self.sqkv = nn.Linear(dim, dim * 3, bias=True)
        self.so = nn.Linear(dim, dim, bias=True)

        self.ln_ffn = nn.LayerNorm(dim)
        hidden = int(round(dim * ffn_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(
        self,
        proto: torch.Tensor,
        h_tokens: torch.Tensor,
        h_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, m_cnt, _ = proto.shape
        n_h = h_tokens.shape[1]

        pn = self.ln_q(proto)
        hn = self.ln_kv(h_tokens)
        q = self.cq(pn).reshape(bsz, m_cnt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.ck(hn).reshape(bsz, n_h, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.cv(hn).reshape(bsz, n_h, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_mask = None
        if h_valid_mask is not None and not h_valid_mask.all():
            safe_mask = h_valid_mask.clone()
            empty_rows = ~safe_mask.any(dim=-1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True
            attn_mask = safe_mask[:, None, None, :].expand(bsz, self.num_heads, m_cnt, n_h)

        c_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        proto = proto + self.co(c_out.permute(0, 2, 1, 3).reshape(bsz, m_cnt, self.dim))

        sn = self.ln_sa(proto)
        qkv = self.sqkv(sn).reshape(bsz, m_cnt, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        sa_out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        proto = proto + self.so(sa_out.permute(0, 2, 1, 3).reshape(bsz, m_cnt, self.dim))

        return proto + self.ffn(self.ln_ffn(proto))


class Stage2HLRetrievalModule(nn.Module):
    """RG4 后第二阶段非局部 H→L 高频检索补偿模块（严格置换等变、掩膜隔离与恒等零初始化）。"""

    def __init__(
        self,
        in_ch: int = 48,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_protos: int = 128,
        num_frames: int = 9,
        max_patch_cols: int = 64,
        ffn_ratio: float = 2.0,
        alpha_h_init_logit: float = -2.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_protos = num_protos
        self.num_frames = num_frames
        self.max_patch_cols = max_patch_cols

        # 1. 8x8 固定物理分块高频差分编码器: [u4, u4 - up(avg_pool_8(u4))] (96 -> 128)
        self.patch_stem = nn.Sequential(
            nn.Conv2d(in_ch * 2, 96, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=True),
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        # 2. 典范序 H 锚点与全局原型字典蒸馏 (M=128)
        self.proto_q = nn.Parameter(torch.empty(1, num_protos, embed_dim))
        self.h_col_embed = nn.Embedding(max_patch_cols, embed_dim)
        self.h_anchor_embed = nn.Embedding(2, embed_dim)  # 0=左端 H 锚点，1=右端 H 锚点
        self.distill_layers = nn.ModuleList([
            _ProtoDistillBlock(dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio)
            for _ in range(2)
        ])

        # 3. 双源（全局 H 字典 + 同行 H token）交叉检索注意力 (2 层)
        self.q_frame_embed = nn.Embedding(num_frames, embed_dim)
        self.q_col_embed = nn.Embedding(max_patch_cols, embed_dim)
        self.cross_layers = nn.ModuleList([
            _CrossAttnFFNBlock(dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_ratio)
            for _ in range(2)
        ])
        self.retrieved_norm = nn.LayerNorm(embed_dim)

        # 4. 三级逐倍上采样还原全分辨率 (128 -> 96 -> 64 -> 48)
        self.up1_conv = nn.Conv2d(embed_dim, 96, kernel_size=3, stride=1, padding=1, bias=True)
        self.up1_dw = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=True)
        self.up2_conv = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.up2_dw = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=True)
        self.up3_conv = nn.Conv2d(64, in_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.up3_dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=True)
        self.up_act = nn.GELU()

        # 5. 逐像素质量非对称阀门与内容门控 + zero-init Conv1x1 出口
        self.content_gate = nn.Sequential(
            nn.Conv2d(in_ch * 2 + 2, in_ch, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
        )
        self.alpha_h_logit = nn.Parameter(torch.tensor(float(alpha_h_init_logit)))
        self.exit_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True)

        self._init_non_zero_weights()
        self.reset_zero_init_branches()

    def _init_non_zero_weights(self):
        trunc_normal_(self.proto_q, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reset_zero_init_branches(self):
        """确保第二阶段检索残差出口严格从恒等映射起点开始。"""
        nn.init.zeros_(self.exit_proj.weight)
        nn.init.zeros_(self.exit_proj.bias)

    def _select_h_anchors(
        self,
        frame_ids: torch.Tensor,
        masked_frame_bool: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按置换无关的典范物理优先级选出至多 2 帧有效 H 参考锚点。

        返回:
          slot_idx: [B, 2] 被选源帧在当前 batch 序列维 (0..T-1) 中的下标
          slot_valid: [B, 2] bool，表示该槽位是否为未屏蔽的有效高质来源帧
          slot_anchor_side: [B, 2] long (0=偏左帧/低 fid，1=偏右帧/高 fid)
        """
        bsz, t_count = frame_ids.shape
        device = frame_ids.device

        # 根据物理帧号计算整帧 H 覆盖比例（0 与 8 为 1.0，1 与 7 为 0.75，4 为 0.0）
        q_frame = _compute_strip_quality_from_ids(frame_ids, grid_w=4).mean(dim=(2, 3))  # [B, T]
        has_h = q_frame > 0.05
        if masked_frame_bool is not None:
            unmasked_h = has_h & (~masked_frame_bool)
        else:
            unmasked_h = has_h

        # 构建确定性优先级：优先未屏蔽且 q 最高帧；同 q 时左帧优先以便输出槽位按 fid 升序排列
        score = torch.where(unmasked_h, q_frame * 100.0 - frame_ids.float() * 1e-2, torch.full_like(q_frame, -1e6))
        top_vals, top_pos = torch.topk(score, k=min(2, t_count), dim=1)
        if t_count == 1:
            top_vals = top_vals.expand(bsz, 2)
            top_pos = top_pos.expand(bsz, 2)

        slot_valid = top_vals > -1e5
        # 若两个槽位均为有效帧，按物理 frame_ids 升序排列确保严格置换不变性
        sel_fids = torch.gather(frame_ids, 1, top_pos)
        need_swap = slot_valid[:, 0] & slot_valid[:, 1] & (sel_fids[:, 0] > sel_fids[:, 1])
        if need_swap.any():
            swapped_pos = top_pos.flip(dims=[1])
            swapped_valid = slot_valid.flip(dims=[1])
            top_pos = torch.where(need_swap.unsqueeze(1), swapped_pos, top_pos)
            slot_valid = torch.where(need_swap.unsqueeze(1), swapped_valid, slot_valid)
            sel_fids = torch.gather(frame_ids, 1, top_pos)

        # 右半侧锚点 fid >= 5 标记为 1，其余标记为 0
        slot_anchor_side = (sel_fids >= 5).long()
        return top_pos, slot_valid, slot_anchor_side

    def _upsample_retrieved(self, tokens_grid: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """tokens_grid: [B*T, D, Hp, Wp] -> [B*T, 48, H, W]。"""
        x = F.interpolate(tokens_grid, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.up_act(self.up1_conv(x))
        x = self.up_act(self.up1_dw(x))

        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.up_act(self.up2_conv(x))
        x = self.up_act(self.up2_dw(x))

        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        x = self.up_act(self.up3_conv(x))
        x = self.up_act(self.up3_dw(x))
        return x

    def forward(
        self,
        u4: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        masked_source_frames: torch.Tensor | Sequence[int] | None = None,
        return_telemetry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """u4: [B, T, 48, H, W] -> 返回校准补充后的 [B, T, 48, H, W]。"""
        bsz, t_count, ch, height, width = u4.shape
        device = u4.device
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'Stage2HLRetrievalModule requires H,W divisible by 8, got ({height}, {width})')

        if frame_ids is None:
            frame_ids = torch.arange(t_count, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, t_count)
        else:
            frame_ids = frame_ids.to(device=device, dtype=torch.long)

        hp, wp = height // 8, width // 8
        u4_flat = u4.reshape(bsz * t_count, ch, height, width)

        # 1. 高频差分 8x8 编码
        low_res = F.avg_pool2d(u4_flat, kernel_size=8, stride=8)
        up_low = F.interpolate(low_res, size=(height, width), mode='bilinear', align_corners=False)
        stem_in = torch.cat([u4_flat, u4_flat - up_low], dim=1)  # [B*T, 96, H, W]
        patch_feat = self.patch_stem(stem_in)  # [B*T, D, Hp, Wp]
        patch_tokens = patch_feat.permute(0, 2, 3, 1).reshape(bsz, t_count, hp, wp, self.embed_dim)
        patch_tokens = self.patch_norm(patch_tokens)

        # 2. 典范序 H 参考源提取与逐 patch 质量过滤
        masked_frame_bool = _masked_frame_flags(frame_ids, masked_source_frames)
        slot_pos, slot_valid, slot_side = self._select_h_anchors(frame_ids, masked_frame_bool)
        sel_fids = torch.gather(frame_ids, 1, slot_pos)  # [B, 2]

        # 计算锚点每个 8x8 物理块的 H 覆盖率，避免 fallback 帧（如 F1/F7）内的 L 块混入 H 词典
        q_patch_h = _compute_pixel_quality_from_ids(sel_fids, width=wp).squeeze(2).squeeze(2) > 0.5  # [B, 2, Wp]
        h_patch_valid = slot_valid.unsqueeze(-1).expand(bsz, 2, wp) & q_patch_h  # [B, 2, Wp]
        has_any_h = slot_valid.any(dim=1)  # [B]

        # 取出 H 锚点 tokens: [B, 2, Hp, Wp, D]
        gather_idx = slot_pos.view(bsz, 2, 1, 1, 1).expand(bsz, 2, hp, wp, self.embed_dim)
        h_grid = torch.gather(patch_tokens, 1, gather_idx)

        # 给 H 锚点加入列位置编码与左/右端编码
        if wp <= self.max_patch_cols:
            col_idx = torch.arange(wp, device=device, dtype=torch.long)
        else:
            col_idx = torch.linspace(0, self.max_patch_cols - 1, wp, device=device).long()
        col_emb_h = self.h_col_embed(col_idx).view(1, 1, 1, wp, self.embed_dim)
        side_emb_h = self.h_anchor_embed(slot_side).view(bsz, 2, 1, 1, self.embed_dim)
        h_grid_pe = h_grid + col_emb_h + side_emb_h

        # 2a. 全图 H tokens -> M=128 全局原型字典 [B, M, D]
        h_all_tokens = h_grid_pe.reshape(bsz, 2 * hp * wp, self.embed_dim)
        h_all_valid = h_patch_valid.unsqueeze(2).expand(bsz, 2, hp, wp).reshape(bsz, 2 * hp * wp)
        proto = self.proto_q.to(dtype=h_all_tokens.dtype).expand(bsz, self.num_protos, self.embed_dim)
        for d_layer in self.distill_layers:
            proto = d_layer(proto, h_all_tokens, h_valid_mask=h_all_valid)

        # 2b. 同行局部 H tokens: [B*Hp, 2*Wp, D]，与全局字典拼接成 256 个双源 key
        row_h_tokens = h_grid_pe.permute(0, 2, 1, 3, 4).reshape(bsz * hp, 2 * wp, self.embed_dim)
        row_h_valid = h_patch_valid.unsqueeze(1).expand(bsz, hp, 2, wp).reshape(bsz * hp, 2 * wp)

        dict_expand = proto.unsqueeze(1).expand(bsz, hp, self.num_protos, self.embed_dim).reshape(bsz * hp, self.num_protos, self.embed_dim)
        dict_valid = has_any_h.unsqueeze(1).expand(bsz, hp).unsqueeze(-1).expand(bsz, hp, self.num_protos).reshape(bsz * hp, self.num_protos)

        kv_tokens = torch.cat([dict_expand, row_h_tokens], dim=1)  # [B*Hp, M + 2*Wp, D]
        kv_valid = torch.cat([dict_valid, row_h_valid], dim=1)

        # 3. 各帧同行 query [B*Hp, T*Wp, D] 双层交叉检索
        q_frame_pe = self.q_frame_embed(frame_ids).view(bsz, t_count, 1, 1, self.embed_dim)
        q_col_pe = self.q_col_embed(col_idx).view(1, 1, 1, wp, self.embed_dim)
        q_grid_pe = patch_tokens + q_frame_pe + q_col_pe
        q_rows = q_grid_pe.permute(0, 2, 1, 3, 4).reshape(bsz * hp, t_count * wp, self.embed_dim)

        for c_layer in self.cross_layers:
            q_rows = c_layer(q_rows, kv_tokens, kv_valid_mask=kv_valid)

        # 按 batch 是否存在有效未屏蔽 H 源进行安全隔离掩膜
        row_batch_valid = has_any_h.unsqueeze(1).expand(bsz, hp).reshape(bsz * hp, 1, 1).to(q_rows.dtype)
        retrieved_rows = self.retrieved_norm(q_rows) * row_batch_valid

        retrieved_grid = (
            retrieved_rows.reshape(bsz, hp, t_count, wp, self.embed_dim)
            .permute(0, 2, 4, 1, 3)
            .reshape(bsz * t_count, self.embed_dim, hp, wp)
        )

        # 4. 三级逐倍上采样至原像素尺度
        up_feat = self._upsample_retrieved(retrieved_grid, target_hw=(height, width))  # [B*T, 48, H, W]

        # 5. 逐像素物理质量非对称阀门与内容门控
        q_pix = _compute_pixel_quality_from_ids(frame_ids, width=width).expand(bsz, t_count, 1, height, width)
        q_pix_flat = q_pix.reshape(bsz * t_count, 1, height, width).to(dtype=u4.dtype)

        # 局部高频能量代理通道，补齐 [u4(48), up_feat(48), q_pix(1), hf_energy(1)] = 98 通道
        local_hf = (u4_flat - F.avg_pool2d(u4_flat, kernel_size=3, stride=1, padding=1)).square().mean(dim=1, keepdim=True)
        local_hf = torch.sqrt(local_hf + 1e-6)

        content_g = torch.sigmoid(self.content_gate(torch.cat([u4_flat, up_feat, q_pix_flat, local_hf], dim=1)))
        alpha_h = torch.sigmoid(self.alpha_h_logit).to(dtype=u4.dtype)
        valve = alpha_h * q_pix_flat + (1.0 - q_pix_flat)
        full_gate = content_g * valve

        delta_raw = self.exit_proj(up_feat * full_gate)
        batch_mask = has_any_h.view(bsz, 1, 1, 1, 1).to(dtype=u4.dtype)
        delta_u4 = delta_raw.reshape(bsz, t_count, ch, height, width) * batch_mask
        u4_out = u4 + delta_u4

        if return_telemetry:
            fg_f = full_gate.reshape(bsz, t_count, ch, height, width).float()
            q_f = q_pix.float()
            l_mask = q_f < 0.1
            h_mask = q_f > 0.9
            l_gate_mean = float(fg_f.mean(dim=2, keepdim=True)[l_mask].mean()) if l_mask.any() else 0.0
            h_gate_mean = float(fg_f.mean(dim=2, keepdim=True)[h_mask].mean()) if h_mask.any() else 0.0
            tele = {
                'stage2_delta_rms': float(delta_u4.detach().float().square().mean().sqrt()),
                'stage2_alpha_h': float(alpha_h.detach().float()),
                'stage2_l_gate_mean': l_gate_mean,
                'stage2_h_gate_mean': h_gate_mean,
            }
            return u4_out, tele
        return u4_out
