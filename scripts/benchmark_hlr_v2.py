"""单卡 24GB RTX 4090 上 UCAN-HLR 训练与全幅推理显存/吞吐探针脚本。"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from basicsr.archs import build_network
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.models import build_model


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


def _make_hlr_train_opt(chunk: int) -> dict:
    return {
        'name': 'probe_hlr',
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': 1,
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
            'train_frame_chunk_size': chunk,
            'eval_frame_batch_size': 1,
        },
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'param_key_g': 'params',
            'resume_state': None,
        },
        'train': {
            'total_iter': 10,
            'warmup_iter': -1,
            'ema_decay': 0.9985,
            'ema_warmup': True,
            'amp_dtype': 'bf16',
            'freeze_backbone_iters': 1,
            'backbone_lr': 1.5e-5,
            'grad_clip_norm': 1.0,
            'l_region_weight': 3.0,
            'f4_frame_weight': 1.5,
            'overlap_consistency_weight': 0.2,
            'grad_loss_weight': 0.1,
            'ssim_loss_weight': 0.05,
            'optim_g': {'type': 'Adam', 'lr': 1.5e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]},
            'scheduler': {'type': 'MultiStepLR', 'milestones': [8], 'gamma': 0.5},
            'pixel_opt': {'type': 'CharbonnierLoss', 'loss_weight': 1.0, 'reduction': 'mean', 'eps': 1e-9},
        },
        'val': {'save_img': False},
    }


def main():
    assert torch.cuda.is_available(), 'CUDA device required'

    net_probe = build_network({
        'type': 'UCANSequence',
        'backbone_opt': _backbone_opt(),
        'token_opt': None,
        'hlr_opt': {'enabled': True},
        'eval_frame_batch_size': 1,
    })
    param_counts = {
        'backbone': int(sum(p.numel() for p in net_probe.backbone.parameters())),
        'stage1_hlr': int(sum(p.numel() for p in net_probe.stage1_hlr.parameters())),
        'stage2_hlr': int(sum(p.numel() for p in net_probe.stage2_hlr.parameters())),
        'total': int(sum(p.numel() for p in net_probe.parameters())),
    }
    print(f'[param_counts] {param_counts}', flush=True)

    # 1. 全幅 512x512 FP32 eval_frame_batch_size=1 推理探针
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    net_eval = net_probe.cuda().eval()
    dummy_eval = torch.rand(1, 9, 1, 512, 512, device='cuda')
    fids = torch.arange(9, dtype=torch.long, device='cuda').unsqueeze(0)
    q_eval = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).cuda()
    eval_times = []
    with torch.no_grad():
        _ = net_eval(dummy_eval, frame_ids=fids, token_quality=q_eval)
        torch.cuda.synchronize()
        for _ in range(3):
            t0 = time.perf_counter()
            out_e = net_eval(dummy_eval, frame_ids=fids, token_quality=q_eval)
            torch.cuda.synchronize()
            eval_times.append((time.perf_counter() - t0) * 1000.0)
            assert torch.isfinite(out_e).all()
    eval_res = {
        'precision': 'fp32',
        'eval_frame_batch_size': 1,
        'seq_ms_mean': float(np.mean(eval_times)),
        'peak_allocated_gib': float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        'peak_reserved_gib': float(torch.cuda.max_memory_reserved() / (1024 ** 3)),
    }
    print(f'[eval_512_fp32] {eval_res}', flush=True)
    del net_eval, net_probe, dummy_eval, out_e
    gc.collect()
    torch.cuda.empty_cache()

    # 2. H=128 BF16 训练吞吐与峰值显存探针（测试 bs=2 chunk=3,2,1 及 bs=3 chunk=2、bs=1 chunk=3）
    configs = [
        (128, 2, 3),
        (128, 2, 2),
        (128, 2, 1),
        (128, 3, 2),
        (128, 1, 3),
    ]
    train_matrix = []
    for height, bs, chunk in configs:
        rec = {
            'height': height,
            'precision': 'bf16',
            'batch_size_per_gpu': bs,
            'chunk': chunk,
            'status': 'ok',
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = None
        try:
            m = build_model(_make_hlr_train_opt(chunk=chunk))
            batch = {
                'lq_seq': torch.rand(bs, 9, 1, height, 512),
                'gt_seq': torch.rand(bs, 9, 1, height, 512),
                'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0).expand(bs, 9),
                'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).expand(bs, 9, 4, 4, 1),
            }
            step_times = []
            # 依次验证冻结步 (1) 与解冻步 (2, 3, 4)
            for step in (1, 2, 3, 4):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.update_learning_rate(step)
                m.feed_data(batch)
                m.optimize_parameters(step)
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
                for lk, lv in m.log_dict.items():
                    if not np.isfinite(lv):
                        raise ValueError(f'Non-finite loss {lk}={lv}')
            sec_per_iter = float(np.mean(step_times[2:]))
            rec['sec_per_iter'] = sec_per_iter
            rec['seqs_per_sec_1gpu'] = float(bs / sec_per_iter)
            rec['seqs_per_sec_2gpu_est'] = float(2.0 * bs / sec_per_iter)
            rec['peak_allocated_gib'] = float(torch.cuda.max_memory_allocated() / (1024 ** 3))
            rec['peak_reserved_gib'] = float(torch.cuda.max_memory_reserved() / (1024 ** 3))
        except torch.cuda.OutOfMemoryError:
            rec['status'] = 'OOM'
        except Exception as exc:
            rec['status'] = f'ERR: {type(exc).__name__}: {exc}'
        finally:
            if m is not None:
                del m
            gc.collect()
            torch.cuda.empty_cache()
        print(
            f"[train_hlr] H={height} bf16 bs={bs} chunk={chunk} -> {rec['status']} "
            + (
                f"| {rec['sec_per_iter']:.3f}s/it | {rec['seqs_per_sec_2gpu_est']:.2f} seq/s(2GPU) "
                f"| alloc={rec['peak_allocated_gib']:.2f}G res={rec['peak_reserved_gib']:.2f}G"
                if rec['status'] == 'ok'
                else ''
            ),
            flush=True,
        )
        train_matrix.append(rec)

    out_dir = Path('docs/UCAN_exp1/diagnostics/throughput_4090_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'device_name': torch.cuda.get_device_name(0),
        'param_counts': param_counts,
        'eval_512x512_fp32_efbs1': eval_res,
        'train_matrix': train_matrix,
    }
    (out_dir / 'hlr_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved HLR benchmark summary to {out_dir / "hlr_summary.json"}')


if __name__ == '__main__':
    main()
