from pathlib import Path
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from basicsr.data import build_dataset, build_dataloader
from basicsr.utils.options import yaml_load


ROOT = Path(__file__).resolve().parents[1]


def test_sequence_frames_and_grayscale(sar_pair):
    dataset = build_dataset(sar_pair)
    assert len(dataset) == 18
    assert [(s[2], s[3]) for s in dataset.samples] == [
        (name, idx) for name in ('sequence_B', 'sequence_A') for idx in range(9)]
    item = dataset[10]
    assert item['sequence_name'] == 'sequence_A' and item['frame_idx'] == 1
    assert item['lq'].shape == item['gt'].shape == (1, 512, 512)
    assert item['lq'].dtype == torch.float32
    for key in ('lq', 'gt'):
        expected = cv2.imread(item[f'{key}_path'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        np.testing.assert_array_equal(item[key].numpy()[0], expected)
    no_split = build_dataset(dict(sar_pair, split_file=None))
    assert no_split[0]['sequence_name'] == 'sequence_A'


def test_paired_crop_and_dp_batch(sar_pair):
    dataset = build_dataset(dict(sar_pair, phase='train'))
    with patch('basicsr.data.sar_single_dataset.random.randint', side_effect=[17, 29]):
        item = dataset[0]
    for key in ('lq', 'gt'):
        original = cv2.imread(item[f'{key}_path'], cv2.IMREAD_GRAYSCALE)
        expected = original[17:145, 29:157].astype(np.float32) / 255
        np.testing.assert_array_equal(item[key].numpy()[0], expected)
    opt = dict(sar_pair, phase='train', batch_size_per_gpu=3, num_worker_per_gpu=0)
    loader = build_dataloader(dataset, opt, num_gpu=2, dist=False, seed=42)
    assert next(iter(loader))['lq'].shape == (6, 1, 128, 128)


def test_invalid_split_and_missing_pair(sar_pair):
    split = Path(sar_pair['split_file'])
    split.write_text('sequence_A\nsequence_A\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        build_dataset(sar_pair)
    split.write_text('../sequence_A\n', encoding='utf-8')
    with pytest.raises(ValueError, match='folder name'):
        build_dataset(sar_pair)
    split.write_text('sequence_A\n', encoding='utf-8')
    (Path(sar_pair['dataroot_gt']) / 'sequence_A/008.png').unlink()
    with pytest.raises(FileNotFoundError, match='Paired frame'):
        build_dataset(sar_pair)


def test_invalid_image_shape(sar_pair):
    path = Path(sar_pair['dataroot_lq']) / 'sequence_B/000.png'
    cv2.imwrite(str(path), np.zeros((256, 256), dtype=np.uint8))
    dataset = build_dataset(sar_pair)
    with pytest.raises(ValueError, match='image shape'):
        dataset[0]


def test_committed_split_and_shared_protocol():
    configs = [yaml_load(str(ROOT / f'options/SAR/train_{arch}_single.yml')) for arch in ('swinir', 'hint')]
    for arch, config in zip(('swinir', 'hint'), configs):
        assert config['datasets']['train']['split_file'] == 'docs/splits/sar_single_seed42/train.txt'
        assert config['datasets']['val']['split_file'] == 'docs/splits/sar_single_seed42/val.txt'
        assert config['datasets']['val']['name'] == 'sar_val'
        assert config['val']['save_img'] is False
        assert config['final_test_opt'] == f'options/SAR/test_{arch}_single.yml'
        test_config = yaml_load(str(ROOT / config['final_test_opt']))
        assert test_config['datasets']['test']['split_file'] == 'docs/splits/sar_single_seed42/test.txt'
        assert test_config['val']['save_img'] is True
    train = (ROOT / 'docs/splits/sar_single_seed42/train.txt').read_text(encoding='utf-8').splitlines()
    val = (ROOT / 'docs/splits/sar_single_seed42/val.txt').read_text(encoding='utf-8').splitlines()
    test = (ROOT / 'docs/splits/sar_single_seed42/test.txt').read_text(encoding='utf-8').splitlines()
    assert len(train) == len(set(train)) == 3688
    assert len(val) == len(set(val)) == 82
    assert len(test) == len(set(test)) == 328
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    expected = sorted(train + val + test)
    random.Random(42).shuffle(expected)
    assert expected[:3688] == train
    held_out = expected[3688:]
    chosen = set(random.Random(42).sample(held_out, 82))
    assert val == [name for name in held_out if name in chosen]
    assert test == [name for name in held_out if name not in chosen]
    data_root = Path(configs[0]['datasets']['train']['dataroot_lq'])
    if data_root.is_dir():
        assert set(train + val + test) == {p.name for p in data_root.iterdir() if p.is_dir()}
