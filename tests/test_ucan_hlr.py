"""Step 3 UCAN-HLR 架构单测：参数预算、恒等零初始化、置换等变性、F4 因果隔离、4 步梯度全贯通与 DDP+BF16 兼容。"""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from basicsr.archs import build_network
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.models import build_model


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _backbone_opt() -> dict:
    return {
        'type': 'UCAN',
        'upscale': 1,
        'in_chans': 1,
        'img_size': 128,
        'embed_dim': 48,
        'window_size': 16,
        'img_range': 1.0,
        'mlp_ratio': 1.0,
        'conv_depth': 5,
        'share': ['N', 'F', 'N', 'F'],
        'mhsa_num_heads': 2,
        'dfl_num_heads': 1,
        'use_checkpoint': False,
        'upsampler': None,
        'resi_connection': '1conv',
        'lksa_gate_gamma': 4.0,
    }


def _make_hlr_net(train_chunk: int = 0, eval_chunk: int = 1) -> torch.nn.Module:
    return build_network({
        'type': 'UCANSequence',
        'backbone_opt': _backbone_opt(),
        'token_opt': None,
        'hlr_opt': {'enabled': True},
        'train_frame_chunk_size': train_chunk,
        'eval_frame_batch_size': eval_chunk,
    })


def _make_dummy_batch(batch_size: int = 1, height: int = 32, width: int = 64, seed: int = 7) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        'lq_seq': torch.rand(batch_size, 9, 1, height, width, generator=gen),
        'gt_seq': torch.rand(batch_size, 9, 1, height, width, generator=gen),
        'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0).expand(batch_size, 9).clone(),
        'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).expand(batch_size, 9, 4, 4, 1).clone(),
    }


def test_hlr_param_budget_and_identity_start():
    """验证总参数严格落在 [2.0M, 3.0M] 内，且冷启零初始化输出与 A_init 骨干完全一致。"""
    torch.manual_seed(42)
    net_base = build_network({
        'type': 'UCANSequence',
        'backbone_opt': _backbone_opt(),
        'token_opt': None,
        'hlr_opt': None,
        'eval_frame_batch_size': 0,
    })
    net_hlr = _make_hlr_net(train_chunk=0, eval_chunk=0)

    bb_params = sum(p.numel() for p in net_hlr.backbone.parameters())
    total_params = sum(p.numel() for p in net_hlr.parameters())
    assert bb_params == 683_817
    assert 2_000_000 <= total_params <= 3_000_000, f'Unexpected total params: {total_params}'

    net_hlr.backbone.load_state_dict(net_base.backbone.state_dict(), strict=True)
    net_base.eval()
    net_hlr.eval()

    batch = _make_dummy_batch(batch_size=1, height=32, width=128)
    with torch.no_grad():
        out_base = net_base(batch['lq_seq'], frame_ids=batch['frame_ids'], token_quality=batch['token_quality'])
        out_hlr, tele = net_hlr(
            batch['lq_seq'],
            frame_ids=batch['frame_ids'],
            token_quality=batch['token_quality'],
            return_telemetry=True,
        )
    assert torch.allclose(out_hlr, out_base, atol=1e-5)
    assert tele['stage1_wb_b_rms'] == pytest.approx(0.0, abs=1e-7)
    assert tele['stage1_delta_freq_rms'] == pytest.approx(0.0, abs=1e-7)
    assert tele['stage1_cons_delta_rms'] == pytest.approx(0.0, abs=1e-7)
    assert tele['stage2_delta_rms'] == pytest.approx(0.0, abs=1e-7)


def _train_four_steps_opt(tmp_path: Path, **train_overrides) -> dict:
    opt = {
        'name': 'test_hlr_model',
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'manual_seed': 42,
        'network_g': {
            'type': 'UCANSequence',
            'backbone_opt': _backbone_opt(),
            'token_opt': None,
            'hlr_opt': {'enabled': True},
            'train_frame_chunk_size': 0,
            'eval_frame_batch_size': 1,
        },
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


def test_four_step_gradient_flow_all_params(tmp_path):
    """验证第 4 步迭代时零初始化出口已打开，Stage1/Stage2（含 proto_q 与 alpha_h_logit）及骨干梯度全贯通。"""
    torch.manual_seed(42)
    opt = _train_four_steps_opt(tmp_path, backbone_lr=1e-4, grad_clip_norm=1.0)
    model = build_model(opt)
    batch = _make_dummy_batch(batch_size=1, height=32, width=64)

    for step in range(1, 5):
        model.update_learning_rate(step)
        model.feed_data(batch)
        model.optimize_parameters(step)

    dead = []
    for name, param in model.net_g.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None or not torch.isfinite(param.grad).all() or float(param.grad.abs().max()) == 0.0:
            dead.append(name)
    assert dead == [], f'Parameters with zero/missing/non-finite grad at step 4: {dead}'


def test_hlr_permutation_equivariance_and_f4_isolation(tmp_path):
    """在非零活跃权重下验证 9 帧置换等变性、eval chunk=1 一致性及 F4 来源全屏蔽因果隔离。"""
    torch.manual_seed(42)
    opt = _train_four_steps_opt(tmp_path, backbone_lr=1e-4)
    model = build_model(opt)
    batch = _make_dummy_batch(batch_size=1, height=32, width=128)
    for step in range(1, 5):
        model.update_learning_rate(step)
        model.feed_data(batch)
        model.optimize_parameters(step)

    net = model.net_g.eval()
    perm = torch.tensor([4, 8, 1, 0, 7, 3, 6, 2, 5], dtype=torch.long)
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        ref_out, tele = net(
            batch['lq_seq'],
            frame_ids=batch['frame_ids'],
            token_quality=batch['token_quality'],
            return_telemetry=True,
        )
        assert tele['stage2_delta_rms'] > 0.0
        assert tele['stage2_l_gate_mean'] > tele['stage2_h_gate_mean']

        perm_out = net(
            batch['lq_seq'][:, perm],
            frame_ids=batch['frame_ids'][:, perm],
            token_quality=batch['token_quality'][:, perm],
        )[:, inv_perm]
        assert torch.allclose(perm_out, ref_out, atol=1e-4)

        # 验证 eval chunk=1 与无 chunk 一致
        net.eval_frame_batch_size = 0
        nochunk_out = net(
            batch['lq_seq'],
            frame_ids=batch['frame_ids'],
            token_quality=batch['token_quality'],
        )
        assert torch.allclose(nochunk_out, ref_out, atol=1e-5)

        # 验证 F4 隔离：屏蔽除 4 以外全部源帧并篡改其余帧像素，F4 输出必须严格不变
        masked_others = [0, 1, 2, 3, 5, 6, 7, 8]
        lq_perturbed = batch['lq_seq'].clone()
        lq_perturbed[:, masked_others] = torch.rand_like(lq_perturbed[:, masked_others])
        iso_base = net(
            batch['lq_seq'],
            frame_ids=batch['frame_ids'],
            token_quality=batch['token_quality'],
            masked_source_frames=masked_others,
        )
        iso_pert = net(
            lq_perturbed,
            frame_ids=batch['frame_ids'],
            token_quality=batch['token_quality'],
            masked_source_frames=masked_others,
        )
        assert torch.allclose(iso_pert[:, 4], iso_base[:, 4], atol=1e-5)


def test_hlr_multi_height_compatibility():
    """验证 H=64 与 H=128（W=128）两种训练高度下前后向均正常，尺度无关结构可直接切换高度。"""
    torch.manual_seed(42)
    net = _make_hlr_net(train_chunk=0, eval_chunk=1)
    for height in (64, 128):
        batch = _make_dummy_batch(batch_size=1, height=height, width=128)
        out = net(batch['lq_seq'], frame_ids=batch['frame_ids'], token_quality=batch['token_quality'])
        assert out.shape == (1, 9, 1, height, 128)
        assert torch.isfinite(out).all()


def _hlr_ddp_worker(rank, rendezvous, out_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        opt = _train_four_steps_opt(
            Path(out_dir),
            amp_dtype='bf16',
            freeze_backbone_iters=1,
            backbone_lr=1e-5,
            ema_decay=0.9985,
            ema_warmup=True,
            grad_clip_norm=1.0,
            l_region_weight=3.0,
            f4_frame_weight=1.5,
            overlap_consistency_weight=0.2,
            grad_loss_weight=0.1,
            ssim_loss_weight=0.05,
        )
        opt['network_g']['train_frame_chunk_size'] = 2
        model = build_model(opt)
        model.net_g = DistributedDataParallel(model.net_g)
        model.net_g._set_static_graph()
        model.opt.update(dist=True, rank=rank, world_size=2)
        batch = _make_dummy_batch(batch_size=2, height=32, width=64)
        for step in (1, 2):
            model.update_learning_rate(step)
            model.feed_data(batch)
            model.optimize_parameters(step)
            assert model.output.dtype == torch.float32
            assert torch.isfinite(model.output).all()
    finally:
        dist.destroy_process_group()


def test_hlr_chunk2_ddp_static_graph_freeze_bf16_bs2(tmp_path):
    """验证 UCAN-HLR 在 chunk=2 + DDP static_graph + freeze_backbone + BF16 + bs=2 下训练完全正常。"""
    rendezvous = (tmp_path / 'hlr_ddp').as_uri()
    mp.spawn(_hlr_ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
