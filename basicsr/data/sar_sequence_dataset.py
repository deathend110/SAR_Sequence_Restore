"""序列化 SAR 配对数据集：读取完整 9 帧序列、同步纵向裁剪并生成精确 H/L 质量来源标签。"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data

from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

# 基于 SAR Dataset Transformer 源码推导的 0-based 逻辑宽度 1536 上精确 H/L 掩膜：
#   full_mask[1..856] 与 [1369..2224] 为 H，结合 sar_image_roi_indices 起始偏移 344，
#   对应 0-based 全局列 [0, 513) 与 [1025, 1536) 为 H，[513, 1025) 为 L。
DEFAULT_LOGIC_LENGTH = 1536
DEFAULT_H_INTERVALS = ((0, 513), (1025, 1536))


def compute_grid_quality_tensor(
    frame_ids: list[int] | tuple[int, ...] | torch.Tensor,
    grid_h: int = 4,
    grid_w: int = 4,
    frame_width: int = 512,
    stride: int = 128,
    h_intervals: tuple[tuple[int, int], ...] = DEFAULT_H_INTERVALS,
) -> torch.Tensor:
    """根据帧编号与横向条带投影到生成器 H 掩膜，计算每个区域 token 的 H 覆盖比例 q_i ∈ [0, 1]。

    返回形状: [T, grid_h, grid_w, 1], float32
    """
    if isinstance(frame_ids, torch.Tensor):
        frame_list = [int(v) for v in frame_ids.tolist()]
    else:
        frame_list = [int(v) for v in frame_ids]

    if frame_width % grid_w != 0:
        raise ValueError(f'frame_width={frame_width} must be divisible by grid_w={grid_w}')
    patch_w = frame_width // grid_w

    t_count = len(frame_list)
    q_tensor = torch.zeros((t_count, grid_h, grid_w, 1), dtype=torch.float32)
    for idx, t in enumerate(frame_list):
        frame_g0 = stride * t
        for k in range(grid_w):
            g0 = frame_g0 + k * patch_w
            g1 = g0 + patch_w
            covered = 0
            for h0, h1 in h_intervals:
                overlap = min(g1, h1) - max(g0, h0)
                if overlap > 0:
                    covered += overlap
            ratio = float(covered) / float(patch_w)
            q_tensor[idx, :, k, 0] = ratio
    return q_tensor


@DATASET_REGISTRY.register()
class SARSequenceDataset(data.Dataset):
    """九帧 1-bit SAR 序列配对数据集（支持确定性配对裁剪计划）。"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt['phase']
        self.crop_height = opt.get('crop_height')
        if self.phase == 'train':
            if self.crop_height is None:
                self.crop_height = 64
            if not isinstance(self.crop_height, int) or not 1 <= self.crop_height <= 512:
                raise ValueError(f'crop_height must be an integer in [1, 512], got {self.crop_height}')
        elif self.crop_height is not None:
            if not isinstance(self.crop_height, int) or not 1 <= self.crop_height <= 512:
                raise ValueError(f'crop_height must be None or in [1, 512], got {self.crop_height}')

        self.grid_h = int(opt.get('grid_h', 4))
        self.grid_w = int(opt.get('grid_w', 4))
        self.deterministic_crop_seed = opt.get('deterministic_crop_seed')
        self.epoch = 0

        root_lq = Path(opt['dataroot_lq'])
        root_gt = Path(opt['dataroot_gt'])
        for root in (root_lq, root_gt):
            if not root.is_dir():
                raise FileNotFoundError(f'Dataset directory not found: {root}')

        split_file = opt.get('split_file')
        if split_file:
            with open(split_file, encoding='utf-8') as stream:
                sequence_names = [line.strip() for line in stream if line.strip()]
            if len(sequence_names) != len(set(sequence_names)):
                raise ValueError(f'Duplicate sequence names in split_file: {split_file}')
            for name in sequence_names:
                if name in ('.', '..') or '/' in name or '\\' in name:
                    raise ValueError(f'Expected a sequence folder name in {split_file}: {name}')
        else:
            names_lq = {p.name for p in root_lq.iterdir() if p.is_dir()}
            names_gt = {p.name for p in root_gt.iterdir() if p.is_dir()}
            if names_lq != names_gt:
                raise ValueError(
                    f'Sequence mismatch: missing in {root_gt}: {sorted(names_lq - names_gt)}; '
                    f'missing in {root_lq}: {sorted(names_gt - names_lq)}'
                )
            sequence_names = sorted(names_lq)

        self.sequences = []
        for sequence_name in sequence_names:
            for root in (root_lq, root_gt):
                if not (root / sequence_name).is_dir():
                    raise FileNotFoundError(f'Sequence directory not found: {root / sequence_name}')
            lq_paths = []
            gt_paths = []
            for frame_idx in range(9):
                filename = f'{frame_idx:03d}.png'
                p_lq = root_lq / sequence_name / filename
                p_gt = root_gt / sequence_name / filename
                for p in (p_lq, p_gt):
                    if not p.is_file():
                        raise FileNotFoundError(f'Paired frame not found: {p}')
                lq_paths.append(str(p_lq))
                gt_paths.append(str(p_gt))
            self.sequences.append((sequence_name, lq_paths, gt_paths))

        # 固定 9 帧的帧编号与静态 H/L 质量标签
        self.base_frame_ids = torch.arange(9, dtype=torch.long)
        self.base_token_quality = compute_grid_quality_tensor(
            self.base_frame_ids, grid_h=self.grid_h, grid_w=self.grid_w
        )

    def set_epoch(self, epoch: int):
        """由外部 sampler 在每轮开始前同步 epoch，用于确定性配对裁剪。"""
        self.epoch = int(epoch)

    def _select_crop_top(self, index: int) -> int:
        if self.crop_height is None or self.crop_height == 512:
            return 0
        max_top = 512 - self.crop_height
        if self.deterministic_crop_seed is not None:
            # 绑定 seed、epoch 与序列索引，保证同 seed 下 A/B/C/D1/D0 的样本裁剪计划逐项一致。
            seed_val = (int(self.deterministic_crop_seed) * 1000003 + self.epoch * 1009 + index) & 0xFFFFFFFFFFFFFFFF
            return random.Random(seed_val).randint(0, max_top)
        return random.randint(0, max_top)

    def __getitem__(self, index):
        sequence_name, lq_paths, gt_paths = self.sequences[index]
        crop_top = self._select_crop_top(index) if self.phase == 'train' or self.crop_height is not None else 0
        crop_height = self.crop_height if self.crop_height is not None else 512

        lq_frames = []
        gt_frames = []
        for p_lq, p_gt in zip(lq_paths, gt_paths):
            img_lq = cv2.imread(p_lq, cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(p_gt, cv2.IMREAD_GRAYSCALE)
            for img, path in ((img_lq, p_lq), (img_gt, p_gt)):
                if img is None:
                    raise ValueError(f'Cannot decode SAR image: {path}')
                if img.shape != (512, 512):
                    raise ValueError(f'Expected image shape (512, 512), got {img.shape}: {path}')
            if crop_height < 512:
                img_lq = img_lq[crop_top:crop_top + crop_height, :]
                img_gt = img_gt[crop_top:crop_top + crop_height, :]
            lq_frames.append(img2tensor(np.ascontiguousarray(img_lq[:, :, None]), bgr2rgb=False).div_(255.0))
            gt_frames.append(img2tensor(np.ascontiguousarray(img_gt[:, :, None]), bgr2rgb=False).div_(255.0))

        lq_seq = torch.stack(lq_frames, dim=0)  # [9, 1, H, 512]
        gt_seq = torch.stack(gt_frames, dim=0)  # [9, 1, H, 512]
        source_group = sequence_name.split('_rstart_')[0]

        return {
            'lq_seq': lq_seq,
            'gt_seq': gt_seq,
            'token_quality': self.base_token_quality.clone(),
            'frame_ids': self.base_frame_ids.clone(),
            'crop_top': crop_top,
            'crop_height': crop_height,
            'sequence_name': sequence_name,
            'source_group': source_group,
            'lq_paths': lq_paths,
            'gt_paths': gt_paths,
        }

    def __len__(self):
        return len(self.sequences)
