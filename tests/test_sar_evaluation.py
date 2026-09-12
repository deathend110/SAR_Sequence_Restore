import csv
import logging
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from basicsr.utils import tensor2img
from scripts.eval_sar import evaluate, METRIC_NAMES


class KnownRestoration:
    """确定性的输出，用于独立核对指标，不依赖随机网络权重。"""
    def feed_data(self, batch):
        self.batch = batch

    def test(self):
        self.output = self.batch['lq'] * 0.9

    def get_current_visuals(self):
        return dict(lq=self.batch['lq'], gt=self.batch['gt'], result=self.output)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_statistics_logging_and_plot(sar_pair, tmp_path, caplog):
    dataset = build_dataset(sar_pair)
    output_dir = tmp_path / 'evaluation'
    with caplog.at_level(logging.INFO, logger='sar_test'):
        means = evaluate(KnownRestoration(), DataLoader(dataset, batch_size=1), output_dir,
                         logging.getLogger('sar_test'))
    records = [r for r in caplog.records if r.name == 'sar_test']
    assert len(records) == 1
    assert {p.name for p in output_dir.iterdir()} == {
        'metrics_per_frame.csv', 'frame_position_summary.csv', 'metrics_per_sequence.csv'}
    assert not list(output_dir.rglob('*.png'))
    fields, rows = read_csv(output_dir / 'metrics_per_frame.csv')
    assert fields == ['sequence_name', 'frame_idx', *METRIC_NAMES]
    assert [(r['sequence_name'], int(r['frame_idx'])) for r in rows] == [
        (name, idx) for name in ('sequence_A', 'sequence_B') for idx in range(9)]
    samples = {(item['sequence_name'], item['frame_idx']): item for item in dataset}
    for row in rows:
        item = samples[row['sequence_name'], int(row['frame_idx'])]
        gt = tensor2img(item['gt'].clone())
        lq = tensor2img(item['lq'].clone())
        restored = tensor2img(item['lq'] * 0.9)
        for metric in ('psnr', 'ssim'):
            opt = dict(type=f'calculate_{metric}', crop_border=0, test_y_channel=False)
            before = calculate_metric(dict(img=lq, img2=gt), opt)
            after = calculate_metric(dict(img=restored, img2=gt), opt)
            assert float(row[f'input_{metric}']) == before
            assert float(row[f'output_{metric}']) == after
            assert float(row[f'delta_{metric}']) == after - before
    _, frames = read_csv(output_dir / 'frame_position_summary.csv')
    _, sequences = read_csv(output_dir / 'metrics_per_sequence.csv')
    assert [int(row['frame_idx']) for row in frames] == list(range(9))
    assert [row['sequence_name'] for row in sequences] == ['sequence_A', 'sequence_B']
    for frame in frames:
        assert int(frame['count']) == 2
        for key in METRIC_NAMES:
            values = [float(r[key]) for r in rows if r['frame_idx'] == frame['frame_idx']]
            np.testing.assert_allclose(float(frame[f'{key}_mean']), np.mean(values))
            np.testing.assert_allclose(float(frame[f'{key}_std']), np.std(values, ddof=0))
    for sequence in sequences:
        assert int(sequence['count']) == 9
        for key in METRIC_NAMES:
            values = [float(r[key]) for r in rows if r['sequence_name'] == sequence['sequence_name']]
            np.testing.assert_allclose(float(sequence[f'{key}_mean']), np.mean(values))
    for key in METRIC_NAMES:
        np.testing.assert_allclose(means[key], np.mean([float(r[key]) for r in rows]))
        assert f'mean {key}: {means[key]:.6f}' in records[0].getMessage()
    subprocess.run([sys.executable, 'scripts/plot_sar_metrics.py', str(output_dir)], check=True)
    for metric in ('psnr', 'ssim'):
        assert (output_dir / f'frame_position_{metric}.png').stat().st_size > 0


def test_saved_restorations_match_metric_pixels(sar_pair, tmp_path):
    dataset = build_dataset(sar_pair)
    evaluate(KnownRestoration(), DataLoader(dataset, batch_size=1), tmp_path,
             logging.getLogger('sar_images'), save_img=True)
    assert len(list((tmp_path / 'restored').rglob('*.png'))) == len(dataset)
    for sample in dataset:
        path = tmp_path / 'restored' / sample['sequence_name'] / f'{sample["frame_idx"]:03d}.png'
        saved = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert saved.shape == (512, 512) and saved.dtype == np.uint8
        np.testing.assert_array_equal(saved, tensor2img(sample['lq'] * 0.9))
