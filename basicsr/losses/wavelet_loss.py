"""面向 SAR 的单层 SWT 子带重建损失，计算约定参考 UCAN 官方实现。

上游版本：b812c0adb8ce70d442973db5c87bab7d1e8c3333，详见
docs/SAR_wavelet_loss_preparation_20261105.md。仅注册损失，不自动启用训练。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

# PyWavelets 1.8.0 的 sym19 分析低通系数；固定系数避免新增运行时依赖。
_SYM19_DEC_LO = (
    5.487732768215838e-07,
    -6.463651303345963e-07,
    -1.1880518269823984e-05,
    8.873312173729286e-06,
    0.0001155392333357879,
    -4.612039600210587e-05,
    -0.000635764515004334,
    0.00015915804768084938,
    0.0021214250281823303,
    -0.0011607032572062486,
    -0.005122205002583014,
    0.007968438320613306,
    0.01579743929567463,
    -0.02265199337824595,
    -0.046635983534938946,
    0.0070155738571741596,
    0.008954591173043624,
    -0.06752505804029409,
    0.10902582508127781,
    0.578144945338605,
    0.7195555257163943,
    0.2582661692372836,
    -0.17659686625203097,
    -0.11624173010739675,
    0.09363084341589714,
    0.08407267627924504,
    -0.016908234861345205,
    -0.02770989693131125,
    0.004319351874894969,
    0.008262236955528255,
    -0.0006179223277983108,
    -0.0017049602611649971,
    0.00012930767650701415,
    0.0002762187768573407,
    -1.6821387029373716e-05,
    -2.8151138661550245e-05,
    2.0623170632395688e-06,
    1.7509367995348687e-06,
)


@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    """单层、不降采样的小波子带 L1，默认直接处理 [0, 1] SAR 灰度值。

    每个输入通道独立变换，输出子带按 UCAN 顺序排列：LL、LH、HL、HH。
    此处 LH 表示宽度低通、高度高通；HL 表示宽度高通、高度低通。
    不执行 RGB→Y 转换，也不隐式放大灰度动态范围。

    Args:
        loss_weight (float): 整体损失权重，默认 1.0。
        loss_weight_ll/lh/hl/hh (float): 各子带权重，默认均为 0.01。
        wavelet (str): 'sym19'（UCAN 使用的 38 tap 滤波器）或 'haar'。
        mode (str): 'reflect'（默认，避免将 patch 两侧相接）或 'periodic'。
        reduction (str): 'none' 返回 NCHW 逐位置加权和；也支持 'mean'、'sum'。
    """

    def __init__(self,
                 loss_weight=1.0,
                 loss_weight_ll=0.01,
                 loss_weight_lh=0.01,
                 loss_weight_hl=0.01,
                 loss_weight_hh=0.01,
                 wavelet='sym19',
                 mode='reflect',
                 reduction='mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        if mode not in ('reflect', 'periodic'):
            raise ValueError(f'Unsupported SWT boundary mode: {mode}.')
        if wavelet not in ('sym19', 'haar'):
            raise ValueError(f'Unsupported wavelet: {wavelet}. Supported ones are sym19 and haar.')
        weights = (loss_weight, loss_weight_ll, loss_weight_lh, loss_weight_hl, loss_weight_hh)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError('SWT loss weights must be finite and nonnegative.')

        self.loss_weight = loss_weight
        self.subband_weights = (loss_weight_ll, loss_weight_lh, loss_weight_hl, loss_weight_hh)
        self.wavelet = wavelet
        self.mode = mode
        self.reduction = reduction

        dec_lo = torch.tensor(_SYM19_DEC_LO if wavelet == 'sym19' else (2**-0.5, 2**-0.5), dtype=torch.float64)
        # 正交镜像关系生成高通；仅将低通归一化为单位直流增益，与 UCAN 一致。
        signs = torch.tensor([(-1)**(i + 1) for i in range(dec_lo.numel())], dtype=torch.float64)
        dec_hi = dec_lo.flip(0) * signs
        filters = torch.stack((dec_lo / dec_lo.sum(), dec_hi)).flip(-1)
        # conv2d 执行互相关，故分析滤波器需翻转；缓冲区随 loss.to(device) 迁移。
        self.register_buffer('row_filters', filters.reshape(2, 1, 1, -1))
        self.register_buffer('col_filters', filters.reshape(2, 1, -1, 1))

    def _pad(self, x, dim):
        size = self.row_filters.shape[-1]
        before, after = size // 2 - 1, size // 2
        if self.mode == 'periodic':
            # 模索引支持奇数尺寸，也支持尺寸小于滤波器长度的输入。
            indices = torch.arange(-before, x.shape[dim] + after, device=x.device) % x.shape[dim]
            return x.index_select(dim, indices)
        if x.shape[dim] <= after:
            raise ValueError(f'{self.wavelet} reflect padding requires each spatial dimension > {after}.')
        pad = (before, after, 0, 0) if dim == 3 else (0, 0, before, after)
        return F.pad(x, pad, mode='reflect')

    def transform(self, x):
        """返回 [N, C, 4, H, W] 系数；保持空间位置，不做阈值去噪或下采样。"""
        if x.ndim != 4 or not x.is_floating_point() or x.numel() == 0:
            raise ValueError('SWT expects a nonempty floating-point NCHW tensor.')
        channels = x.shape[1]
        row = self.row_filters.to(x).repeat(channels, 1, 1, 1)
        col = self.col_filters.to(x).repeat(2 * channels, 1, 1, 1)
        horizontal = F.conv2d(self._pad(x, 3), row, groups=channels)
        bands = F.conv2d(self._pad(horizontal, 2), col, groups=2 * channels)
        return bands.reshape(x.shape[0], channels, 4, x.shape[2], x.shape[3])

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError('SWT prediction and target must have identical shapes; broadcasting is not supported.')
        if pred.dtype != target.dtype or pred.device != target.device:
            raise ValueError('SWT prediction and target must have the same dtype and device.')
        # 线性变换满足 W(pred)-W(target)=W(pred-target)，只变换残差可减少计算和中间张量。
        residual_bands = self.transform(pred - target).abs()
        weights = residual_bands.new_tensor(self.subband_weights).view(1, 1, 4, 1, 1)
        loss = (residual_bands * weights).sum(dim=2)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return self.loss_weight * loss
