"""使用临时实验检查训练完成后的自动测试，不读取正式最终测试集。"""
from copy import deepcopy
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import yaml

from basicsr.models.swinir_model import SwinIRModel
from basicsr.train import train_pipeline


ROOT = Path(__file__).resolve().parents[1]


def write_options(sar_pair, tmp_path, num_gpu):
    # 三个独立 sequence 名称，全部来自测试夹具，不接触正式数据。
    for quality in ('lq', 'gt'):
        root = Path(sar_pair[f'dataroot_{quality}'])
        shutil.copytree(root / 'sequence_A', root / 'sequence_C')
    opt = yaml.safe_load((ROOT / 'options/SAR/train_swinir_single.yml').read_text(encoding='utf-8'))
    opt.update(name='sar_final_flow', num_gpu=num_gpu)
    opt['path']['experiments_root'] = str(tmp_path / 'experiments')
    opt['network_g'].update(embed_dim=6, depths=[2], num_heads=[2], drop_path_rate=0)
    opt['train']['total_iter'] = 2
    opt['val']['val_freq'] = 1
    opt['logger'].update(print_freq=1, save_checkpoint_freq=1, use_tb_logger=False)
    for phase, sequence in [('train', 'sequence_B'), ('val', 'sequence_A')]:
        split = tmp_path / f'{phase}.txt'
        split.write_text(sequence + '\n', encoding='utf-8')
        opt['datasets'][phase].update(
            dataroot_lq=sar_pair['dataroot_lq'], dataroot_gt=sar_pair['dataroot_gt'], split_file=str(split))
    opt['datasets']['train'].update(batch_size_per_gpu=1, num_worker_per_gpu=0)
    test = yaml.safe_load((ROOT / 'options/SAR/test_swinir_single.yml').read_text(encoding='utf-8'))
    test['network_g'] = deepcopy(opt['network_g'])
    test['num_gpu'] = min(num_gpu, 1)
    test['path']['results_root'] = str(tmp_path / 'results')
    # 此路径故意无效，自动测试必须使用刚保存的实际实验检查点覆盖它。
    test['path']['pretrain_network_g'] = str(tmp_path / 'unused.pth')
    split = tmp_path / 'test.txt'
    split.write_text('sequence_C\n', encoding='utf-8')
    test['datasets']['test'].update(
        dataroot_lq=sar_pair['dataroot_lq'], dataroot_gt=sar_pair['dataroot_gt'], split_file=str(split))
    test_path = tmp_path / 'test.yml'
    test_path.write_text(yaml.safe_dump(test), encoding='utf-8')
    opt['final_test_opt'] = str(test_path)
    train_path = tmp_path / 'train.yml'
    train_path.write_text(yaml.safe_dump(opt), encoding='utf-8')
    return train_path


def test_interrupted_training_does_not_run_final_test(sar_pair, tmp_path, monkeypatch):
    config = write_options(sar_pair, tmp_path, num_gpu=0)
    monkeypatch.setattr(sys, 'argv', ['train.py', '-opt', str(config)])
    launch = Mock()
    monkeypatch.setattr('basicsr.train.subprocess.run', launch)
    original = SwinIRModel.optimize_parameters

    def interrupt(model, step):
        if step == 2:
            raise RuntimeError('test interruption')
        original(model, step)

    monkeypatch.setattr(SwinIRModel, 'optimize_parameters', interrupt)
    with pytest.raises(RuntimeError, match='test interruption'):
        train_pipeline(str(ROOT))
    launch.assert_not_called()
    models = tmp_path / 'experiments/sar_final_flow/models'
    assert (models / 'net_g_1.pth').is_file()
    assert not (models / 'net_g_latest.pth').exists()
    assert not (tmp_path / 'results').exists()


def test_training_without_final_test_option(sar_pair, tmp_path, monkeypatch):
    config = write_options(sar_pair, tmp_path, num_gpu=0)
    opt = yaml.safe_load(config.read_text(encoding='utf-8'))
    del opt['final_test_opt']
    config.write_text(yaml.safe_dump(opt), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['train.py', '-opt', str(config)])
    launch = Mock()
    monkeypatch.setattr('basicsr.train.subprocess.run', launch)
    train_pipeline(str(ROOT))
    launch.assert_not_called()
    assert (tmp_path / 'experiments/sar_final_flow/models/net_g_latest.pth').is_file()


@pytest.mark.skipif(os.environ.get('SAR_GPU_SMOKE') != '1', reason='Set SAR_GPU_SMOKE=1 for two-GPU verification.')
def test_ddp_completion_runs_final_test_once(sar_pair, tmp_path):
    config = write_options(sar_pair, tmp_path, num_gpu=2)
    result = subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        'basicsr/train.py', '-opt', str(config), '--launcher', 'pytorch',
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    console = result.stdout + result.stderr
    assert console.count('Start final SAR test') == 1
    assert console.index('Save the latest model.') < console.index('Start final SAR test')
    assert console.count('SAR summary |') == 1
    output = tmp_path / 'results/sar_final_flow/net_g_latest/sar_test'
    paths = sorted((output / 'restored').rglob('*.png'))
    assert [path.relative_to(output).as_posix() for path in paths] == [
        f'restored/sequence_C/{index:03d}.png' for index in range(9)]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image.shape == (512, 512) and image.dtype == np.uint8
    with (output / 'metrics_per_frame.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert {row['sequence_name'] for row in rows} == {'sequence_C'}
    assert (output / 'metrics_per_sequence.csv').is_file()
    assert (output / 'frame_position_summary.csv').is_file()
