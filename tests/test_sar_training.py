"""显式启用的完整模型双卡检查：SAR_GPU_SMOKE=1，仅运行短流程。"""
from copy import deepcopy
import gc
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils.options import yaml_load


@pytest.mark.skipif(os.environ.get('SAR_GPU_SMOKE') != '1', reason='Set SAR_GPU_SMOKE=1 for full two-GPU verification.')
@pytest.mark.parametrize('arch', ['hint'])
def test_full_dp_train_validate_resume(arch, tmp_path):
    assert torch.cuda.device_count() == 2
    root = Path(__file__).resolve().parents[1]
    opt = yaml_load(str(root / f'options/SAR/train_{arch}_single.yml'))
    opt.update(is_train=True, dist=False, rank=0, world_size=1)
    for key in ('models', 'training_states', 'visualization'):
        path = tmp_path / key
        path.mkdir()
        opt['path'][key] = str(path)
    # 用小 milestone 实测恢复后的调度边界，完整网络和 batch 不缩小。
    opt['train']['scheduler']['milestones'] = [3]
    for phase in ('train', 'val'):
        opt['datasets'][phase]['phase'] = phase
        opt['datasets'][phase]['split_file'] = str(root / opt['datasets'][phase]['split_file'])
    train = build_dataset(opt['datasets']['train'])
    batch = next(iter(DataLoader(train, batch_size=6, num_workers=0)))
    val_opt = deepcopy(opt['datasets']['val'])
    split = tmp_path / 'val.txt'
    split.write_text(Path(val_opt['split_file']).read_text().splitlines()[0] + '\n')
    val_opt['split_file'] = str(split)
    val_loader = DataLoader(build_dataset(val_opt), batch_size=1, num_workers=0)
    # 只压缩架构打印，不屏蔽错误或改变官方训练/验证流程。
    logging.getLogger('basicsr').setLevel(logging.WARNING)
    model = build_model(deepcopy(opt))
    assert isinstance(model.net_g, torch.nn.DataParallel)
    for device in (0, 1):
        torch.cuda.reset_peak_memory_stats(device)
    for step in (1, 2):
        model.update_learning_rate(step, warmup_iter=-1)
        model.feed_data(batch)
        model.optimize_parameters(step)
        assert math.isfinite(model.log_dict['l_pix'])
    train_peak = [torch.cuda.max_memory_allocated(d) / 2**30 for d in (0, 1)]
    model.validation(val_loader, current_iter=2, tb_logger=None, save_img=False)
    assert model.net_g.training
    assert set(model.metric_results) == {'psnr', 'ssim'}
    for value in model.metric_results.values():
        assert torch.isfinite(torch.tensor(value))
    assert not list((tmp_path / 'visualization').rglob('*.png'))
    # 验证后继续训练，保存模型、优化器和 scheduler 原生状态。
    model.update_learning_rate(3, warmup_iter=-1)
    model.feed_data(batch)
    model.optimize_parameters(3)
    assert math.isfinite(model.log_dict['l_pix'])
    model.save(0, 3)
    resume_state = torch.load(tmp_path / 'training_states/3.state', map_location='cpu', weights_only=True)
    expected_weights = {k: v.detach().cpu().clone() for k, v in model.get_bare_model(model.net_g).state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    opt['path']['pretrain_network_g'] = str(tmp_path / 'models/net_g_3.pth')
    restored = build_model(deepcopy(opt))
    restored.resume_training(resume_state)
    for key, value in restored.get_bare_model(restored.net_g).state_dict().items():
        torch.testing.assert_close(value.cpu(), expected_weights[key], rtol=0, atol=0)
    for actual, expected in zip(restored.optimizers, resume_state['optimizers']):
        assert actual.state_dict()['param_groups'] == expected['param_groups']
        for param_id, state in actual.state_dict()['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value.cpu(), expected['state'][param_id][key].cpu(), rtol=0, atol=0)
    assert restored.schedulers[0].state_dict() == resume_state['schedulers'][0]
    restored.update_learning_rate(4, warmup_iter=-1)
    assert restored.optimizer_g.param_groups[0]['lr'] == pytest.approx(opt['train']['optim_g']['lr'] * 0.5)
    restored.feed_data(batch)
    restored.optimize_parameters(4)
    assert math.isfinite(restored.log_dict['l_pix'])
    report = dict(arch=arch, train_peak_allocated_gib=train_peak,
                  restored_loss=float(restored.log_dict['l_pix']), validation_frames=9)
    print('SAR_GPU_RESULT', json.dumps(report), flush=True)


@pytest.mark.skipif(os.environ.get('SAR_GPU_SMOKE') != '1', reason='Set SAR_GPU_SMOKE=1 for full two-GPU verification.')
def test_swinir_ddp_train_validate_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(root / 'tests/swinir_ddp_worker.py'), '--output', str(tmp_path),
    ], cwd=root, check=True)


@pytest.mark.skipif(os.environ.get('SAR_GPU_SMOKE') != '1', reason='Set SAR_GPU_SMOKE=1 for full two-GPU verification.')
def test_hint_ddp_train_validate_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(root / 'tests/hint_ddp_worker.py'), '--output', str(tmp_path),
    ], cwd=root, check=True)
