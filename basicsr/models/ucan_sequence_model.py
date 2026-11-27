"""UCAN 九帧序列训练与验证模型包装（支持从 A_init 加载骨干、EMA、差异化学习率、冻结热身及面向 V 谷/高频的复合损失）。"""

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from basicsr.archs import build_network
from basicsr.data.sar_sequence_dataset import DEFAULT_H_INTERVALS
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sequence_validation import SequenceDistributedValidationMixin
from .sr_model import SRModel


def _create_ssim_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    """构造与 BasicSR 口径一致的 11x11 二维高斯 SSIM 卷积核。"""
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window_2d = torch.outer(gauss_1d, gauss_1d)
    return window_2d.view(1, 1, window_size, window_size)


@MODEL_REGISTRY.register()
class UCANSequenceModel(SequenceDistributedValidationMixin, SRModel):
    """九帧联合恢复模型：向后兼容等权 Charbonnier 损失，可选启用 EMA、骨干冻结热身、差异化学习率与 L 区/高频/一致性损失。"""

    def __init__(self, opt):
        super().__init__(opt)
        # 若使用 train_frame_chunk_size 对共享骨干重复执行多次 checkpoint，开启 static_graph 防止 DDP 报 ready twice
        if self.opt.get('dist') and isinstance(self.net_g, DistributedDataParallel):
            bare = self.get_bare_model(self.net_g)
            if getattr(bare, 'train_frame_chunk_size', 0) > 0:
                self.net_g._set_static_graph()

        backbone_path = self.opt['path'].get('pretrain_backbone_g')
        full_path = self.opt['path'].get('pretrain_network_g')
        if backbone_path is not None and full_path is None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            ckpt = torch.load(backbone_path, map_location='cpu', weights_only=True)
            state = ckpt[param_key] if (param_key is not None and param_key in ckpt) else ckpt
            bare = self.get_bare_model(self.net_g)
            bare.load_backbone_state_dict(state, strict=True)
            if hasattr(self, 'net_g_ema'):
                # 骨干热启动后同步初始化 EMA 权重，避免从随机初始化滞后追赶
                self.model_ema(0)
            logger = get_root_logger()
            logger.info(f'Loaded A_init backbone into UCANSequence from {backbone_path}.')

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        logger = get_root_logger()

        self.ema_decay = float(train_opt.get('ema_decay', 0))
        self.ema_warmup = bool(train_opt.get('ema_warmup', False))
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay} (warmup={self.ema_warmup})')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            loaded_ema = False
            if load_path is not None:
                ckpt = torch.load(load_path, map_location='cpu', weights_only=True)
                if isinstance(ckpt, dict) and 'params_ema' in ckpt:
                    self.load_network(
                        self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema'
                    )
                    loaded_ema = True
            if not loaded_ema:
                # 预训练权重不含 params_ema（如 init_C_s42.pth）时直接从 net_g 同步，避免 KeyError
                self.model_ema(0)
            self.net_g_ema.eval()

        # 冻结热身、梯度裁剪与可选复合损失超参（默认全部保持旧版等价行为）
        self.freeze_backbone_iters = int(train_opt.get('freeze_backbone_iters', 0))
        self.grad_clip_norm = float(train_opt.get('grad_clip_norm', 0.0))
        self.l_region_weight = float(train_opt.get('l_region_weight', 1.0))
        self.f4_frame_weight = float(train_opt.get('f4_frame_weight', 1.0))
        self.overlap_consistency_weight = float(train_opt.get('overlap_consistency_weight', 0.0))
        self.grad_loss_weight = float(train_opt.get('grad_loss_weight', 0.0))
        self.ssim_loss_weight = float(train_opt.get('ssim_loss_weight', 0.0))

        bare = self.get_bare_model(self.net_g)
        self._backbone_params = [p for k, p in bare.named_parameters() if k.startswith('backbone.')]
        self._weight_cache = {}
        self._ssim_window_cache = {}

        if self.freeze_backbone_iters > 0:
            logger.info(f'Freeze backbone updates for first {self.freeze_backbone_iters} iters (DDP static_graph safe).')
        if any(
            (
                self.l_region_weight != 1.0,
                self.f4_frame_weight != 1.0,
                self.overlap_consistency_weight > 0.0,
                self.grad_loss_weight > 0.0,
                self.ssim_loss_weight > 0.0,
            )
        ):
            logger.info(
                'Composite loss enabled: '
                f'l_region_weight={self.l_region_weight}, f4_frame_weight={self.f4_frame_weight}, '
                f'overlap_w={self.overlap_consistency_weight}, grad_w={self.grad_loss_weight}, '
                f'ssim_w={self.ssim_loss_weight}'
            )

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_cfg = deepcopy(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type')
        base_lr = float(optim_cfg.pop('lr'))
        backbone_lr = train_opt.get('backbone_lr', None)
        logger = get_root_logger()

        if backbone_lr is None:
            optim_params = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning(f'Params {k} will not be optimized.')
            self.optimizer_g = self.get_optimizer(optim_type, optim_params, lr=base_lr, **optim_cfg)
        else:
            backbone_params, other_params = [], []
            for k, v in self.net_g.named_parameters():
                if not v.requires_grad:
                    logger.warning(f'Params {k} will not be optimized.')
                    continue
                if k.startswith('backbone.') or k.startswith('module.backbone.'):
                    backbone_params.append(v)
                else:
                    other_params.append(v)
            param_groups = []
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': float(backbone_lr)})
            if other_params:
                param_groups.append({'params': other_params, 'lr': base_lr})
            logger.info(
                f'Differential LR: backbone_lr={float(backbone_lr):.3e} ({len(backbone_params)} tensors), '
                f'other_lr={base_lr:.3e} ({len(other_params)} tensors).'
            )
            self.optimizer_g = self.get_optimizer(optim_type, param_groups, lr=base_lr, **optim_cfg)
        self.optimizers.append(self.optimizer_g)

    def _get_canonical_weights(self, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """构造并缓存标准 9 帧的 H/L 空间权重图 [9, 1, 1, W]，均值严格归一化为 1.0。"""
        key = (width, str(device), str(dtype))
        if key in self._weight_cache:
            return self._weight_cache[key]

        weights = torch.ones((9, 1, 1, width), dtype=dtype, device=device)
        if self.l_region_weight != 1.0:
            cols = torch.arange(width, device=device, dtype=torch.float32)
            scale = 512.0 / float(width)
            for t in range(9):
                g_cols = 128.0 * float(t) + cols * scale
                is_h = torch.zeros(width, dtype=torch.bool, device=device)
                for h0, h1 in DEFAULT_H_INTERVALS:
                    is_h |= (g_cols >= float(h0)) & (g_cols < float(h1))
                weights[t, 0, 0, ~is_h] = self.l_region_weight
        if self.f4_frame_weight != 1.0:
            weights[4] = weights[4] * self.f4_frame_weight

        weights = weights / weights.mean()
        self._weight_cache[key] = weights
        return weights

    def _build_batch_weight_flat(
        self, bsz: int, t_count: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """按 batch 内 frame_ids 生成 [B*T, 1, H, W] 像素权重图；全为 1.0 时返回 None。"""
        if self.l_region_weight == 1.0 and self.f4_frame_weight == 1.0:
            return None
        canonical = self._get_canonical_weights(width, device, dtype)  # [9, 1, 1, W]
        if self.frame_ids is not None:
            w_seq = canonical[self.frame_ids]  # [B, T, 1, 1, W]
        else:
            w_seq = canonical[:t_count].unsqueeze(0).expand(bsz, t_count, 1, 1, width)
        # 按当前 batch 再次归一化，保证平均权重严格为 1.0
        w_seq = w_seq / w_seq.mean()
        return w_seq.expand(bsz, t_count, 1, height, width).reshape(bsz * t_count, 1, height, width)

    def _compute_grad_loss(
        self, pred_flat: torch.Tensor, gt_flat: torch.Tensor, weight_flat: torch.Tensor | None
    ) -> torch.Tensor:
        """水平与垂直一阶梯度 L1 损失（保护高频散射体边缘与峰值）。"""
        dx_diff = torch.abs((pred_flat[:, :, :, 1:] - pred_flat[:, :, :, :-1]) - (gt_flat[:, :, :, 1:] - gt_flat[:, :, :, :-1]))
        dy_diff = torch.abs((pred_flat[:, :, 1:, :] - pred_flat[:, :, :-1, :]) - (gt_flat[:, :, 1:, :] - gt_flat[:, :, :-1, :]))
        if weight_flat is not None:
            wx = weight_flat[:, :, :, 1:]
            wy = weight_flat[:, :, 1:, :]
            lx = (dx_diff * wx).sum() / wx.sum().clamp_min(1e-12)
            ly = (dy_diff * wy).sum() / wy.sum().clamp_min(1e-12)
        else:
            lx = dx_diff.mean()
            ly = dy_diff.mean()
        return 0.5 * (lx + ly)

    def _compute_ssim_loss(
        self, pred_flat: torch.Tensor, gt_flat: torch.Tensor, bsz: int, t_count: int
    ) -> torch.Tensor:
        """可微 11x11 高斯窗 SSIM 损失（1 - SSIM），支持按 F4 权重对帧加权。"""
        height, width = pred_flat.shape[-2], pred_flat.shape[-1]
        win_sz = min(11, height, width)
        if win_sz % 2 == 0:
            win_sz = max(3, win_sz - 1)
        key = (win_sz, str(pred_flat.device), str(pred_flat.dtype))
        if key not in self._ssim_window_cache:
            self._ssim_window_cache[key] = _create_ssim_window(win_sz, 1.5, pred_flat.device, pred_flat.dtype)
        window = self._ssim_window_cache[key]

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu1 = F.conv2d(pred_flat, window, padding=0)
        mu2 = F.conv2d(gt_flat, window, padding=0)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred_flat * pred_flat, window, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(gt_flat * gt_flat, window, padding=0) - mu2_sq
        sigma12 = F.conv2d(pred_flat * gt_flat, window, padding=0) - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den.clamp_min(1e-12)
        per_frame_loss = 1.0 - ssim_map.mean(dim=(1, 2, 3))  # [B*T]

        if self.f4_frame_weight != 1.0:
            if self.frame_ids is not None:
                fw = torch.where(self.frame_ids == 4, self.f4_frame_weight, 1.0).to(per_frame_loss.dtype)
            else:
                fw = torch.ones((bsz, t_count), dtype=per_frame_loss.dtype, device=per_frame_loss.device)
                if t_count > 4:
                    fw[:, 4] = self.f4_frame_weight
            fw = (fw / fw.mean()).reshape(-1)
            return (per_frame_loss * fw).mean()
        return per_frame_loss.mean()

    def feed_data(self, data):
        self.lq = data['lq_seq'].to(self.device)  # [B, 9, 1, H, W]
        if 'gt_seq' in data:
            self.gt = data['gt_seq'].to(self.device)
        self.frame_ids = data['frame_ids'].to(self.device) if 'frame_ids' in data else None
        self.token_quality = data['token_quality'].to(self.device) if 'token_quality' in data else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, frame_ids=self.frame_ids, token_quality=self.token_quality)

        bsz, t_count, ch, height, width = self.output.shape
        pred_flat = self.output.reshape(bsz * t_count, ch, height, width)
        gt_flat = self.gt.reshape(bsz * t_count, ch, height, width)
        weight_flat = self._build_batch_weight_flat(bsz, t_count, height, width, pred_flat.device, pred_flat.dtype)

        l_total = 0
        loss_dict = {}
        if self.cri_pix:
            l_pix = self.cri_pix(pred_flat, gt_flat, weight=weight_flat)
            l_total = l_total + l_pix
            loss_dict['l_pix'] = l_pix

        if self.overlap_consistency_weight > 0.0 and t_count > 1:
            pred_ordered = self.output
            if self.frame_ids is not None:
                # 若输入帧发生重排，先按物理帧序排回再计算相邻滑窗重叠一致性
                order = torch.argsort(self.frame_ids, dim=1)
                if not torch.equal(order, torch.arange(t_count, device=order.device).unsqueeze(0).expand_as(order)):
                    gather_idx = order.view(bsz, t_count, 1, 1, 1).expand_as(self.output)
                    pred_ordered = torch.gather(self.output, dim=1, index=gather_idx)
            step_w = max(1, width // 4)
            overlap_diff = pred_ordered[:, :-1, :, :, step_w:] - pred_ordered[:, 1:, :, :, :width - step_w]
            l_overlap = self.overlap_consistency_weight * torch.sqrt(overlap_diff * overlap_diff + 1e-9).mean()
            l_total = l_total + l_overlap
            loss_dict['l_overlap'] = l_overlap

        if self.grad_loss_weight > 0.0:
            l_grad = self.grad_loss_weight * self._compute_grad_loss(pred_flat, gt_flat, weight_flat)
            l_total = l_total + l_grad
            loss_dict['l_grad'] = l_grad

        if self.ssim_loss_weight > 0.0:
            l_ssim = self.ssim_loss_weight * self._compute_ssim_loss(pred_flat, gt_flat, bsz, t_count)
            l_total = l_total + l_ssim
            loss_dict['l_ssim'] = l_ssim

        l_total.backward()

        # 冻结期内清零骨干梯度而非修改 requires_grad，确保 DDP static_graph 安全
        if current_iter <= self.freeze_backbone_iters:
            for p in self._backbone_params:
                p.grad = None

        if self.grad_clip_norm > 0.0:
            active_params = [p for p in self.net_g.parameters() if p.grad is not None]
            if active_params:
                torch.nn.utils.clip_grad_norm_(active_params, self.grad_clip_norm)

        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            eff_decay = (
                min(self.ema_decay, (1.0 + float(current_iter)) / (10.0 + float(current_iter)))
                if self.ema_warmup
                else self.ema_decay
            )
            self.model_ema(decay=eff_decay)

    def test(self):
        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        if isinstance(net, DistributedDataParallel):
            net = self.get_bare_model(net)
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                self.output = self._run_sequence_inference(
                    net, self.lq, frame_ids=self.frame_ids, token_quality=self.token_quality
                )
        finally:
            net.train(was_training)
