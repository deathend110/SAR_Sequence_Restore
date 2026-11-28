"""Step 2 序列模型训练升级单测：冻结热身、差异化学习率、安全 EMA 初始化/预热及面向 V 谷/高频的复合损失。"""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.models import build_model
from scripts.init_paired_weights import build_group_network_opt
from scripts.prune_checkpoints import prune_experiment


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _make_v2_opt(tmp_path: Path, **train_overrides) -> dict:
    opt = {
        'name': 'test_v2_model',
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'manual_seed': 42,
        'network_g': build_group_network_opt('C', lksa_gate_gamma=4.0),
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'param_key_g': 'params',
            'resume_state': None,
            'experiments_root': str(tmp_path),
            'models': str(tmp_path / 'models'),
            'training_states': str(tmp_path / 'training_states'),
            'visualization': str(tmp_path / 'visualization'),
        },
        'train': {
            'total_iter': 10,
            'warmup_iter': -1,
            'ema_decay': 0,
            'optim_g': {'type': 'Adam', 'lr': 1e-3, 'weight_decay': 0, 'betas': [0.9, 0.99]},
            'scheduler': {'type': 'MultiStepLR', 'milestones': [8], 'gamma': 0.5},
            'pixel_opt': {'type': 'CharbonnierLoss', 'loss_weight': 1.0, 'reduction': 'mean', 'eps': 1e-9},
        },
        'val': {
            'save_img': False,
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False},
            },
        },
    }
    opt['train'].update(train_overrides)
    for d in ('models', 'training_states', 'visualization'):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return opt


def _make_dummy_batch(batch_size: int = 1, height: int = 32, width: int = 64) -> dict:
    torch.manual_seed(7)
    return {
        'lq_seq': torch.rand(batch_size, 9, 1, height, width),
        'gt_seq': torch.rand(batch_size, 9, 1, height, width),
        'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0).expand(batch_size, 9),
        'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).expand(batch_size, 9, 4, 4, 1),
    }


def test_freeze_backbone_and_differential_lr(tmp_path):
    """验证冻结期内骨干逐比特不变且 Adam 动量为空，解冻后按 backbone_lr 正常更新。"""
    opt = _make_v2_opt(
        tmp_path,
        freeze_backbone_iters=2,
        backbone_lr=1e-5,
        grad_clip_norm=1.0,
    )
    model = build_model(deepcopy(opt))
    assert len(model.optimizer_g.param_groups) == 2
    assert model.optimizer_g.param_groups[0]['lr'] == pytest.approx(1e-5)
    assert model.optimizer_g.param_groups[1]['lr'] == pytest.approx(1e-3)

    bb_init = {
        k: v.detach().clone()
        for k, v in model.net_g.named_parameters()
        if k.startswith('backbone.')
    }
    ti_init = {
        k: v.detach().clone()
        for k, v in model.net_g.named_parameters()
        if k.startswith('token_interaction.')
    }

    batch = _make_dummy_batch()
    # iter 1, 2 处于冻结期：骨干不变、动量为空；token_interaction 正常更新
    for step in (1, 2):
        model.update_learning_rate(step)
        model.feed_data(batch)
        model.optimize_parameters(step)
        for k, p in model.net_g.named_parameters():
            if k.startswith('backbone.'):
                torch.testing.assert_close(p, bb_init[k], rtol=0, atol=0)
                assert p not in model.optimizer_g.state

    ti_changed = any(
        (p - ti_init[k]).abs().max().item() > 0.0
        for k, p in model.net_g.named_parameters()
        if k.startswith('token_interaction.')
    )
    assert ti_changed

    # iter 3 解冻后：骨干开始更新并写入 Adam 状态
    model.update_learning_rate(3)
    model.feed_data(batch)
    model.optimize_parameters(3)
    bb_changed = any(
        (p - bb_init[k]).abs().max().item() > 0.0
        for k, p in model.net_g.named_parameters()
        if k.startswith('backbone.')
    )
    assert bb_changed


def test_ema_safe_init_without_params_ema_and_warmup(tmp_path):
    """验证从不含 params_ema 的权重启动 EMA 不报错，第 0 步严格一致且支持预热平滑。"""
    # 先保存一份只含 'params' 的权重（模拟 init_C_s42.pth）
    opt_pre = _make_v2_opt(tmp_path / 'pre')
    m_pre = build_model(deepcopy(opt_pre))
    init_ckpt = tmp_path / 'init_params_only.pth'
    torch.save({'params': m_pre.net_g.state_dict()}, init_ckpt)

    opt = _make_v2_opt(tmp_path / 'run', ema_decay=0.999, ema_warmup=True)
    opt['path']['pretrain_network_g'] = str(init_ckpt)
    model = build_model(deepcopy(opt))

    # 第 0 步 net_g_ema 与 net_g 严格一致
    for p, p_ema in zip(model.net_g.parameters(), model.net_g_ema.parameters()):
        torch.testing.assert_close(p_ema, p, rtol=0, atol=0)

    # 执行 1 步后，由于 ema_warmup 生效（iter=1 时 eff_decay = 2/11 < 0.999），EMA 权重会明显向 net_g 靠拢且保持有限
    batch = _make_dummy_batch()
    model.update_learning_rate(1)
    model.feed_data(batch)
    model.optimize_parameters(1)
    diffs = [
        (p_ema - p_init).abs().max().item()
        for p_ema, p_init in zip(model.net_g_ema.parameters(), m_pre.net_g.parameters())
    ]
    assert max(diffs) > 0.0
    for p_ema in model.net_g_ema.parameters():
        assert torch.isfinite(p_ema).all()


def test_composite_loss_weights_and_gradients(tmp_path):
    """验证 L 区倾斜加权（均值严格=1）、重叠一致性、梯度损失与 SSIM 损失数值有限且梯度畅通。"""
    opt = _make_v2_opt(
        tmp_path,
        l_region_weight=2.5,
        f4_frame_weight=1.5,
        overlap_consistency_weight=0.1,
        grad_loss_weight=0.1,
        ssim_loss_weight=0.05,
    )
    model = build_model(deepcopy(opt))
    batch = _make_dummy_batch(height=32, width=512)
    model.feed_data(batch)

    w_flat = model._build_batch_weight_flat(1, 9, 32, 512, torch.device('cpu'), torch.float32)
    assert w_flat.shape == (9, 1, 32, 512)
    assert w_flat.mean().item() == pytest.approx(1.0, rel=1e-6)
    # 验证 F4 中心比 F0 的 H 区权重高（2.5 * 1.5 = 3.75 倍）
    assert (w_flat[4, 0, 0, 200] / w_flat[0, 0, 0, 100]).item() == pytest.approx(3.75, rel=1e-5)

    model.update_learning_rate(1)
    model.optimize_parameters(1)
    for k in ('l_pix', 'l_overlap', 'l_grad', 'l_ssim'):
        assert k in model.log_dict
        assert torch.isfinite(torch.tensor(model.log_dict[k]))
        assert model.log_dict[k] >= 0.0


def test_backward_compatibility_with_v1_loss_and_prune_f4(tmp_path):
    """验证默认关闭新选项时与原 Charbonnier 行为一致，且 prune_checkpoints 支持按 f4_psnr 保留。"""
    opt_v2_default = _make_v2_opt(tmp_path / 'def')
    model = build_model(deepcopy(opt_v2_default))
    batch = _make_dummy_batch()
    model.feed_data(batch)
    assert model._build_batch_weight_flat(1, 9, 32, 64, torch.device('cpu'), torch.float32) is None
    model.optimize_parameters(1)
    assert set(model.log_dict.keys()) == {'l_pix'}

    exp_dir = tmp_path / 'prune_f4'
    (exp_dir / 'models').mkdir(parents=True)
    (exp_dir / 'training_states').mkdir(parents=True)
    for it in (3000, 6000, 9000, 12000):
        (exp_dir / f'models/net_g_{it}.pth').write_bytes(b'x' * 8)
        (exp_dir / f'training_states/{it}.state').write_bytes(b'y' * 8)
    (exp_dir / 'val_history.json').write_text(
        json.dumps([
            {'iter': 3000, 'metrics': {'psnr': 21.0, 'ssim': 0.70}, 'per_frame': {'psnr': [21.2]*4 + [20.8] + [21.2]*4}},
            {'iter': 6000, 'metrics': {'psnr': 21.3, 'ssim': 0.71}, 'per_frame': {'psnr': [21.4]*4 + [21.0] + [21.4]*4}},
            {'iter': 9000, 'metrics': {'psnr': 21.2, 'ssim': 0.70}, 'per_frame': {'psnr': [21.1]*4 + [21.15] + [21.1]*4}},
            {'iter': 12000, 'metrics': {'psnr': 21.1, 'ssim': 0.705}, 'per_frame': {'psnr': [21.1]*4 + [20.9] + [21.1]*4}},
        ]),
        encoding='utf-8',
    )
    manifest = prune_experiment(exp_dir)
    assert manifest['best_psnr']['iter'] == 6000
    assert manifest['best_f4_psnr']['iter'] == 9000
    assert set(manifest['kept_iters']) == {6000, 9000, 12000}
    assert (exp_dir / 'models/net_g_best_f4.pth').is_symlink()


def _freeze_ddp_worker(rank, rendezvous, out_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        opt = _make_v2_opt(
            Path(out_dir),
            freeze_backbone_iters=1,
            backbone_lr=1e-5,
            ema_decay=0.999,
            ema_warmup=True,
            grad_clip_norm=1.0,
            overlap_consistency_weight=0.1,
        )
        opt['network_g']['train_frame_chunk_size'] = 3
        model = build_model(deepcopy(opt))
        model.net_g = DistributedDataParallel(model.net_g)
        model.net_g._set_static_graph()
        model.opt.update(dist=True, rank=rank, world_size=2)
        batch = _make_dummy_batch()
        # step 1 (冻结) 与 step 2 (解冻) 均应在 static_graph 下顺利执行
        for step in (1, 2):
            model.update_learning_rate(step)
            model.feed_data(batch)
            model.optimize_parameters(step)
    finally:
        dist.destroy_process_group()


def test_freeze_static_graph_ddp_safe(tmp_path):
    """验证在 chunk=3 + DDP static_graph 下冻结热身与解冻切换不报任何图变化异常。"""
    rendezvous = (tmp_path / 'v2_ddp').as_uri()
    mp.spawn(_freeze_ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)


def test_bf16_amp_and_multi_batch(tmp_path):
    """验证 batch_size=2 下复合损失与权重图正常，且启用 amp_dtype='bf16' 时训练/验证正常且梯度有限。"""
    opt = _make_v2_opt(
        tmp_path,
        amp_dtype='bf16',
        l_region_weight=2.5,
        f4_frame_weight=1.5,
        overlap_consistency_weight=0.2,
        grad_loss_weight=0.1,
        ssim_loss_weight=0.05,
    )
    opt['val']['use_bf16_val'] = True
    if torch.cuda.is_available():
        opt['num_gpu'] = 1
    model = build_model(deepcopy(opt))
    assert model.amp_dtype == torch.bfloat16
    assert model.use_bf16_val is True

    batch = _make_dummy_batch(batch_size=2, height=32, width=64)
    model.update_learning_rate(1)
    model.feed_data(batch)
    model.optimize_parameters(1)
    assert model.output.dtype == torch.float32
    for lk in ('l_pix', 'l_overlap', 'l_grad', 'l_ssim'):
        assert torch.isfinite(torch.tensor(model.log_dict[lk]))
    for p in model.net_g.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    model.test()
    assert model.output.shape == (2, 9, 1, 32, 64)
    assert model.output.dtype == torch.float32
    assert torch.isfinite(model.output).all()


def test_make_exp_dirs_skips_pretrain_backbone(tmp_path):
    """验证 make_exp_dirs 与路径展开将 pretrain_backbone_g 视为文件路径而非待建目录。"""
    from basicsr.utils import make_exp_dirs

    backbone_file = tmp_path / 'net_g_best_psnr.pth'
    backbone_file.write_bytes(b'dummy')
    exp_root = tmp_path / 'exp_hlr'
    opt = {
        'is_train': True,
        'rank': 0,
        'path': {
            'experiments_root': str(exp_root),
            'models': str(exp_root / 'models'),
            'training_states': str(exp_root / 'training_states'),
            'log': str(exp_root),
            'visualization': str(exp_root / 'visualization'),
            'pretrain_backbone_g': str(backbone_file),
            'strict_load_g': True,
            'param_key_g': 'params',
            'resume_state': None,
        },
    }
    make_exp_dirs(opt)
    assert backbone_file.is_file()
    assert (exp_root / 'models').is_dir()

