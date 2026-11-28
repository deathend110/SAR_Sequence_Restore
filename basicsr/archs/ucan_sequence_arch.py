"""UCAN 序列包装网络：保持 RG1->RG2 与 RG3->RG4 原共享接口，在 RG2 后通过纯张量重排插入跨帧 Token 模块。"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from .cross_frame_token import CrossFrameTokenInteraction
from .hl_retrieval_token import Stage1CalibrationModule, Stage2HLRetrievalModule
from .ucan_arch import UCAN


@ARCH_REGISTRY.register()
class UCANSequence(nn.Module):
    """九帧 SAR 序列恢复网络（Group A 单帧、B 帧内、C 跨帧、D 质量条件化及 Step 3 UCAN-HLR 统一封装）。"""

    def __init__(
        self,
        backbone_opt: dict,
        token_opt: dict | None = None,
        hlr_opt: dict | None = None,
        checkpoint_rg_pair: bool = False,
        eval_frame_batch_size: int = 1,
        train_frame_chunk_size: int = 0,
    ):
        super().__init__()
        self.backbone = UCAN(**backbone_opt)
        self.checkpoint_rg_pair = bool(checkpoint_rg_pair)
        self.eval_frame_batch_size = int(eval_frame_batch_size) if eval_frame_batch_size is not None else 0
        self.train_frame_chunk_size = int(train_frame_chunk_size) if train_frame_chunk_size is not None else 0

        if token_opt is not None and token_opt.get('enabled', True):
            t_cfg = {k: v for k, v in token_opt.items() if k != 'enabled'}
            t_cfg.setdefault('in_ch', backbone_opt.get('embed_dim', 48))
            self.token_interaction = CrossFrameTokenInteraction(**t_cfg)
            # 在 UCAN 初始化之后再显式确保零初始化分支从恒等起点出发
            self.token_interaction.reset_zero_init_branches()
        else:
            self.token_interaction = None

        self.stage1_hlr = None
        self.stage2_hlr = None
        if hlr_opt is not None and hlr_opt.get('enabled', True):
            if self.token_interaction is not None:
                raise ValueError('token_opt and hlr_opt cannot both be enabled simultaneously')
            base_ch = backbone_opt.get('embed_dim', 48)
            s1_cfg = dict(hlr_opt.get('stage1_opt') or {})
            if hlr_opt.get('stage1_enabled', True) and s1_cfg.get('enabled', True):
                s1_cfg.pop('enabled', None)
                s1_cfg.setdefault('in_ch', base_ch)
                self.stage1_hlr = Stage1CalibrationModule(**s1_cfg)
                self.stage1_hlr.reset_zero_init_branches()

            s2_cfg = dict(hlr_opt.get('stage2_opt') or {})
            if hlr_opt.get('stage2_enabled', True) and s2_cfg.get('enabled', True):
                s2_cfg.pop('enabled', None)
                s2_cfg.setdefault('in_ch', base_ch)
                self.stage2_hlr = Stage2HLRetrievalModule(**s2_cfg)
                self.stage2_hlr.reset_zero_init_branches()

    def load_backbone_state_dict(self, state_dict: dict, strict: bool = True):
        """从单帧 UCAN 或 UCANSequence checkpoint 中精确加载骨干权重（绝不重置新增模块的零初始化）。"""
        cleaned = {}
        has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())
        extra_prefixes = ('token_interaction.', 'stage1_hlr.', 'stage2_hlr.')
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if has_backbone_prefix:
                if key.startswith('backbone.'):
                    cleaned[key[len('backbone.'):]] = v
            else:
                if not key.startswith(extra_prefixes):
                    cleaned[key] = v
        return self.backbone.load_state_dict(cleaned, strict=strict)

    def _run_rg_pair(self, rg_first: nn.Module, rg_second: nn.Module, feat: torch.Tensor, x_size: tuple[int, int]):
        """按 RG1->RG2 或 RG3->RG4 配对执行，可选将共享 attention 封装在重计算单元内部以节约显存。"""
        if self.checkpoint_rg_pair and self.training and torch.is_grad_enabled():
            def _pair_fn(x_in: torch.Tensor) -> torch.Tensor:
                out1, attn_map, qk_map = rg_first(x_in, x_size)
                return rg_second(out1, x_size, attn_map, qk_map)

            return torch.utils.checkpoint.checkpoint(_pair_fn, feat, use_reentrant=False)

        out1, attn_map, qk_map = rg_first(feat, x_size)
        return rg_second(out1, x_size, attn_map, qk_map)

    def _encode_chunk(self, x_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """单帧或帧块前半段前向：conv_first -> patch_embed -> rg_1 -> rg_2。"""
        bb = self.backbone
        xf = bb.conv_first(x_chunk)
        xs = (xf.shape[2], xf.shape[3])
        ft = bb.pos_drop(bb.patch_embed(xf))
        out1, attn_map, qk_map = bb.rg_1(ft, xs)
        ft = bb.rg_2(out1, xs, attn_map, qk_map)
        return xf, ft

    def _decode_body_chunk(self, feat_chunk: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """单帧或帧块后半段主体前向：rg_3 -> rg_4 -> norm -> patch_unembed。"""
        bb = self.backbone
        out3, attn_map, qk_map = bb.rg_3(feat_chunk, x_size)
        ft = bb.rg_4(out3, x_size, attn_map, qk_map)
        return bb.patch_unembed(bb.norm(ft), x_size)

    def _reconstruct_chunk(self, u4_chunk: torch.Tensor, xf_chunk: torch.Tensor, x_norm_chunk: torch.Tensor) -> torch.Tensor:
        """单帧或帧块末尾图像残差重建：conv_after_body + x_first -> conv_last + x_norm。"""
        bb = self.backbone
        res = bb.conv_after_body(u4_chunk) + xf_chunk
        return x_norm_chunk + bb.conv_last(res)

    def _decode_chunk(self, feat_chunk: torch.Tensor, xf_chunk: torch.Tensor, x_norm_chunk: torch.Tensor) -> torch.Tensor:
        """单帧或帧块后半段完整前向（保持旧版单 chunk 接口向后兼容）。"""
        xs = (xf_chunk.shape[2], xf_chunk.shape[3])
        u4_chunk = self._decode_body_chunk(feat_chunk, xs)
        return self._reconstruct_chunk(u4_chunk, xf_chunk, x_norm_chunk)

    def forward(
        self,
        lq_seq: torch.Tensor,
        frame_ids: torch.Tensor | None = None,
        token_quality: torch.Tensor | None = None,
        masked_source_frames: torch.Tensor | Sequence[int] | None = None,
        return_telemetry: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        lq_seq: [B, T, 1, H, W]
        返回: [B, T, 1, H, W]
        """
        if lq_seq.dim() != 5:
            raise ValueError(f'Expected 5D input [B, T, C, H, W], got {tuple(lq_seq.shape)}')
        bsz, t_count, ch_in, h_ori, w_ori = lq_seq.shape
        x = lq_seq.reshape(bsz * t_count, ch_in, h_ori, w_ori)

        bb = self.backbone
        # 1. 与 UCAN.forward 完全一致的镜像 padding 与均值/量程规范化
        mod = bb.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        bb.mean = bb.mean.type_as(x)
        x_norm = (x - bb.mean) * bb.img_range

        chunk_bt = bsz * t_count
        use_train_chunk = (
            self.training
            and torch.is_grad_enabled()
            and self.train_frame_chunk_size > 0
            and self.train_frame_chunk_size < chunk_bt
        )
        use_eval_chunk = (
            (not self.training)
            and (not torch.is_grad_enabled())
            and self.eval_frame_batch_size > 0
            and self.eval_frame_batch_size < chunk_bt
        )

        # 2. 浅层特征与 UCAN 前半段（RG1 -> RG2）
        if use_train_chunk:
            step = self.train_frame_chunk_size
            xf_list, ft_list = [], []
            for s_idx in range(0, chunk_bt, step):
                e_idx = min(s_idx + step, chunk_bt)
                xf_c, ft_c = torch.utils.checkpoint.checkpoint(
                    self._encode_chunk, x_norm[s_idx:e_idx], use_reentrant=False
                )
                xf_list.append(xf_c)
                ft_list.append(ft_c)
            x_first = torch.cat(xf_list, dim=0)
            feat = torch.cat(ft_list, dim=0)
            x_size = (x_first.shape[2], x_first.shape[3])
        elif use_eval_chunk:
            step = self.eval_frame_batch_size
            xf_list, ft_list = [], []
            for s_idx in range(0, chunk_bt, step):
                e_idx = min(s_idx + step, chunk_bt)
                xf_c, ft_c = self._encode_chunk(x_norm[s_idx:e_idx])
                xf_list.append(xf_c)
                ft_list.append(ft_c)
            x_first = torch.cat(xf_list, dim=0)
            feat = torch.cat(ft_list, dim=0)
            x_size = (x_first.shape[2], x_first.shape[3])
        else:
            x_first = bb.conv_first(x_norm)
            x_size = (x_first.shape[2], x_first.shape[3])
            feat = bb.pos_drop(bb.patch_embed(x_first))  # [B*T, HW, C]
            feat = self._run_rg_pair(bb.rg_1, bb.rg_2, feat, x_size)

        # 3. 纯张量转置/重排并执行 RG2 后跨帧交互（旧版 token_interaction 或 Step 3 stage1_hlr）
        telemetry = {}
        stage1_mod = self.token_interaction if self.token_interaction is not None else self.stage1_hlr
        if stage1_mod is not None:
            c_dim = feat.shape[-1]
            u_spatial = feat.transpose(1, 2).reshape(bsz, t_count, c_dim, x_size[0], x_size[1])
            inter_out = stage1_mod(
                u_spatial,
                frame_ids=frame_ids,
                token_quality=token_quality,
                masked_source_frames=masked_source_frames,
                return_telemetry=return_telemetry,
            )
            if return_telemetry:
                u_updated, tele1 = inter_out
                telemetry.update(tele1)
            else:
                u_updated = inter_out
            feat = u_updated.reshape(bsz * t_count, c_dim, x_size[0] * x_size[1]).transpose(1, 2).contiguous()

        # 4. UCAN 后半段（RG3 -> RG4）、可选 Step 3 Stage2 H->L 检索与图像重建
        if self.stage2_hlr is None:
            if use_train_chunk:
                step = self.train_frame_chunk_size
                out_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    out_c = torch.utils.checkpoint.checkpoint(
                        self._decode_chunk,
                        feat[s_idx:e_idx],
                        x_first[s_idx:e_idx],
                        x_norm[s_idx:e_idx],
                        use_reentrant=False,
                    )
                    out_list.append(out_c)
                out = torch.cat(out_list, dim=0)
            elif use_eval_chunk:
                step = self.eval_frame_batch_size
                out_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    out_list.append(self._decode_chunk(feat[s_idx:e_idx], x_first[s_idx:e_idx], x_norm[s_idx:e_idx]))
                out = torch.cat(out_list, dim=0)
            else:
                feat = self._run_rg_pair(bb.rg_3, bb.rg_4, feat, x_size)
                feat = bb.patch_unembed(bb.norm(feat), x_size)
                out = self._reconstruct_chunk(feat, x_first, x_norm)
        else:
            c_dim = feat.shape[-1]
            if use_train_chunk:
                step = self.train_frame_chunk_size
                u4_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    u4_c = torch.utils.checkpoint.checkpoint(
                        self._decode_body_chunk,
                        feat[s_idx:e_idx],
                        x_size,
                        use_reentrant=False,
                    )
                    u4_list.append(u4_c)
                u4_spatial = torch.cat(u4_list, dim=0)
            elif use_eval_chunk:
                step = self.eval_frame_batch_size
                u4_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    u4_list.append(self._decode_body_chunk(feat[s_idx:e_idx], x_size))
                u4_spatial = torch.cat(u4_list, dim=0)
            else:
                ft = self._run_rg_pair(bb.rg_3, bb.rg_4, feat, x_size)
                u4_spatial = bb.patch_unembed(bb.norm(ft), x_size)

            u4_seq = u4_spatial.reshape(bsz, t_count, c_dim, x_size[0], x_size[1])
            s2_out = self.stage2_hlr(
                u4_seq,
                frame_ids=frame_ids,
                masked_source_frames=masked_source_frames,
                return_telemetry=return_telemetry,
            )
            if return_telemetry:
                u4_seq, tele2 = s2_out
                telemetry.update(tele2)
            else:
                u4_seq = s2_out
            u4_spatial = u4_seq.reshape(chunk_bt, c_dim, x_size[0], x_size[1])

            if use_train_chunk:
                step = self.train_frame_chunk_size
                out_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    out_c = torch.utils.checkpoint.checkpoint(
                        self._reconstruct_chunk,
                        u4_spatial[s_idx:e_idx],
                        x_first[s_idx:e_idx],
                        x_norm[s_idx:e_idx],
                        use_reentrant=False,
                    )
                    out_list.append(out_c)
                out = torch.cat(out_list, dim=0)
            elif use_eval_chunk:
                step = self.eval_frame_batch_size
                out_list = []
                for s_idx in range(0, chunk_bt, step):
                    e_idx = min(s_idx + step, chunk_bt)
                    out_list.append(self._reconstruct_chunk(u4_spatial[s_idx:e_idx], x_first[s_idx:e_idx], x_norm[s_idx:e_idx]))
                out = torch.cat(out_list, dim=0)
            else:
                out = self._reconstruct_chunk(u4_spatial, x_first, x_norm)
        out = out / bb.img_range + bb.mean
        out = out[..., :h_ori * bb.upscale, :w_ori * bb.upscale]
        out_seq = out.reshape(bsz, t_count, ch_in, out.shape[-2], out.shape[-1])

        if return_telemetry:
            return out_seq, telemetry
        return out_seq
