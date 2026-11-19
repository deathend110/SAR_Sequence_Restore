"""在双卡 RTX 4090 (CUDA_VISIBLE_DEVICES=0,1) 上实测序列训练显存、单步耗时与整帧验证耗时（指导书第 12.2、12.4 节）。"""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.losses import build_loss
from basicsr.models import build_model
from scripts.init_paired_weights import build_group_network_opt


def benchmark_train_modes(rank: int, world_size: int, device: torch.device) -> list[dict]:
    """对 A, B, C, D1 测试分帧重计算配置的单步耗时、峰值显存与数值一致性（带 OOM 容错）。"""
    groups = ['A', 'B', 'C', 'D1']
    # (mode_name, train_frame_chunk_size, use_checkpoint, checkpoint_rg_pair)
    modes = [
        ('chunk3_noblk', 3, False, False),
        ('chunk3_blk', 3, True, False),
        ('chunk1_noblk', 1, False, False),
    ]
    criterion = build_loss({'type': 'CharbonnierLoss', 'loss_weight': 1.0, 'reduction': 'mean', 'eps': 1e-9}).to(device)
    torch.manual_seed(42 + rank)
    x = torch.rand(1, 9, 1, 64, 512, device=device)
    gt = torch.rand(1, 9, 1, 64, 512, device=device)
    fids = torch.arange(9, dtype=torch.long, device=device).unsqueeze(0)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).to(device)

    results = []
    for g in groups:
        ref_out = None
        ref_grad_norm = None
        for mode_name, chunk_sz, use_blk_ckpt, use_pair_ckpt in modes:
            net_opt = build_group_network_opt(g, lksa_gate_gamma=4.0, train_frame_chunk_size=chunk_sz)
            net_opt['backbone_opt']['use_checkpoint'] = use_blk_ckpt
            net_opt['checkpoint_rg_pair'] = use_pair_ckpt

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dist.barrier()

            ddp_net = net = opt = None
            oom_local = 0
            max_out_diff = None
            rel_grad_diff = None
            step_times = []
            peak_alloc = None
            peak_res = None

            try:
                torch.manual_seed(1234)
                net = build_network(net_opt).to(device)
                n_params = sum(p.numel() for p in net.parameters())
                ddp_net = DistributedDataParallel(net, device_ids=[rank])
                if chunk_sz > 0:
                    ddp_net._set_static_graph()
                opt = torch.optim.Adam(ddp_net.parameters(), lr=1e-4, betas=(0.9, 0.99))

                # 数值一致性与梯度验证
                ddp_net.train()
                opt.zero_grad(set_to_none=True)
                out_chk = ddp_net(x, frame_ids=fids, token_quality=q)
                loss_chk = criterion(out_chk.reshape(-1, 1, 64, 512), gt.reshape(-1, 1, 64, 512))
                loss_chk.backward()
                grad_norm = float(np.sqrt(sum(float(p.grad.detach().double().pow(2).sum()) for p in net.parameters() if p.grad is not None)))
                if ref_out is None:
                    ref_out = out_chk.detach().clone()
                    ref_grad_norm = grad_norm
                    max_out_diff = 0.0
                    rel_grad_diff = 0.0
                else:
                    max_out_diff = float((out_chk.detach() - ref_out).abs().max())
                    rel_grad_diff = abs(grad_norm - ref_grad_norm) / max(ref_grad_norm, 1e-12)
                opt.step()

                # 预热 3 步，正式测 10 步
                torch.cuda.reset_peak_memory_stats(device)
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    o = ddp_net(x, frame_ids=fids, token_quality=q)
                    l = criterion(o.reshape(-1, 1, 64, 512), gt.reshape(-1, 1, 64, 512))
                    l.backward()
                    opt.step()

                torch.cuda.synchronize(device)
                for _ in range(10):
                    t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    o = ddp_net(x, frame_ids=fids, token_quality=q)
                    l = criterion(o.reshape(-1, 1, 64, 512), gt.reshape(-1, 1, 64, 512))
                    l.backward()
                    opt.step()
                    torch.cuda.synchronize(device)
                    step_times.append(time.perf_counter() - t0)

                peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                peak_res = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
            except torch.cuda.OutOfMemoryError:
                oom_local = 1

            oom_flag = torch.tensor([oom_local], dtype=torch.int32, device=device)
            dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)

            if oom_flag.item() > 0:
                if rank == 0:
                    results.append({
                        'group': g,
                        'mode': mode_name,
                        'train_frame_chunk_size': chunk_sz,
                        'use_checkpoint': use_blk_ckpt,
                        'checkpoint_rg_pair': use_pair_ckpt,
                        'oom': True,
                    })
            else:
                stats_t = torch.tensor([float(np.mean(step_times)), peak_alloc, peak_res], device=device)
                dist.all_reduce(stats_t, op=dist.ReduceOp.MAX)
                if rank == 0:
                    results.append({
                        'group': g,
                        'mode': mode_name,
                        'train_frame_chunk_size': chunk_sz,
                        'use_checkpoint': use_blk_ckpt,
                        'checkpoint_rg_pair': use_pair_ckpt,
                        'oom': False,
                        'params': n_params,
                        'step_time_s_mean': float(stats_t[0].item()),
                        'peak_alloc_gib': float(stats_t[1].item()),
                        'peak_reserved_gib': float(stats_t[2].item()),
                        'max_out_diff_vs_ref': max_out_diff,
                        'rel_grad_norm_diff': rel_grad_diff,
                    })

            del ddp_net, net, opt
            torch.cuda.empty_cache()
    return results


def benchmark_val_ddp(rank: int, world_size: int, device: torch.device) -> dict:
    """在真实新验证集 val.txt (270 序列 = 2430 帧, 512x512) 上实测双卡 DDP 验证耗时与显存。"""
    data_root = '/data/cjz/SAR_Transformer/dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64'
    ds_opt = {
        'name': 'sar_val_scene',
        'type': 'SARSequenceDataset',
        'phase': 'val',
        'dataroot_lq': f'{data_root}/LQ',
        'dataroot_gt': f'{data_root}/GT',
        'split_file': 'docs/UCAN_exp1/splits/scene_split_v1/val.txt',
    }
    dataset = build_dataset(ds_opt)
    loader = DataLoader(dataset, batch_size=1)
    model_opt = {
        'name': 'cap_bench_val',
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': world_size,
        'is_train': False,
        'dist': True,
        'rank': rank,
        'world_size': world_size,
        'network_g': build_group_network_opt('C', lksa_gate_gamma=4.0),
        'path': {'pretrain_network_g': None, 'strict_load_g': True, 'param_key_g': 'params'},
        'val': {
            'save_img': False,
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False},
            },
        },
    }
    model = build_model(deepcopy(model_opt))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    t0 = time.perf_counter()
    model.validation(loader, current_iter=0, tb_logger=None, save_img=False)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    peak_res = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
    t_stats = torch.tensor([elapsed, peak_alloc, peak_res], device=device)
    dist.all_reduce(t_stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        return {
            'val_sequences': len(dataset),
            'val_frames': len(dataset) * 9,
            'elapsed_s': float(t_stats[0].item()),
            'peak_alloc_gib': float(t_stats[1].item()),
            'peak_reserved_gib': float(t_stats[2].item()),
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='docs/UCAN_exp1/diagnostics/capacity')
    parser.add_argument('--skip-val', action='store_true', help='跳过 270 序列验证测速并复用已有结果')
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    assert world_size == 2, f'Expected 2-GPU DDP, got world_size={world_size}'

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl')

    try:
        out_dir = Path(args.out_dir)
        summary_path = out_dir / 'summary.json'
        prev_payload = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
        train_res = benchmark_train_modes(rank, world_size, device)
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            print('=== TRAIN BENCHMARK ===\n' + json.dumps(train_res, ensure_ascii=False, indent=2), flush=True)
        if args.skip_val and 'val_270seq_512x512_ddp' in prev_payload:
            val_res = prev_payload['val_270seq_512x512_ddp']
        else:
            val_res = benchmark_val_ddp(rank, world_size, device)
        if rank == 0:
            payload = {
                'gpus': [torch.cuda.get_device_name(i) for i in range(world_size)],
                'train_B1_T9_H64_W512_FP32': train_res,
                'val_270seq_512x512_ddp': val_res,
            }
            if 'train_B1_T9_H64_W512_FP32' in prev_payload:
                payload['prev_whole_batch_oom_records'] = prev_payload['train_B1_T9_H64_W512_FP32']
            summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
