"""SAR 测试使用可辨识轴向的灰度图，不依赖训练产物。"""
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sar_pair(tmp_path):
    y, x = np.indices((512, 512))
    for name, offset in [('sequence_B', 20), ('sequence_A', 0)]:
        for quality in ('LQ', 'GT'):
            (tmp_path / quality / name).mkdir(parents=True)
        for idx in range(9):
            gt = ((x + 2 * y + idx + offset) % 120 + 40).astype(np.uint8)
            lq = gt + 10
            lq[0, :] = 240
            for quality, img in [('LQ', lq), ('GT', gt)]:
                assert cv2.imwrite(str(tmp_path / quality / name / f'{idx:03d}.png'), img)
    split = tmp_path / 'split.txt'
    split.write_text('sequence_B\nsequence_A\n', encoding='utf-8')
    return dict(name='sar_fixture', type='SARSingleDataset', phase='test',
                dataroot_lq=str(tmp_path / 'LQ'), dataroot_gt=str(tmp_path / 'GT'),
                split_file=str(split), gt_size=128)
