"""单卡 24GB RTX 4090 训练/验证吞吐、显存上限与 BF16 数值精度一致性基准脚本。"""

import gc
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs import build_network  # noqa: E402
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor  # noqa: E402
from basicsr.metrics import calculate_metric  # noqa: E402
from basicsr.models import build_model  # noqa: E402
from basicsr.utils import img2tensor, tensor2img  # noqa: E402
from basicsr.utils.options import yaml_load  # noqa: E402


def benchmark_train_matrix(base_opt: dict) -> list[dict]:
    """实测不同 (H, amp_dtype, batch_size, train_frame_chunk_size) 的耗时与峰值显存。"""
    matrix = []
    configs = [
        # H=64 组合
        (64, None, 1, 3),
        (64, None, 2, 3),
        (64, None, 2, 6),
        (64, None, 3, 3),
        (64, None, 4, 6),
        (64, 'bf16', 1, 3),
        (64, 'bf16', 2, 3),
        (64, 'bf16', 2, 6),
        (64, 'bf16', 3, 3),
        (64, 'bf16', 4, 6),
        # H=128 组合
        (128, None, 1, 1),
        (128, None, 1, 3),
        (128, None, 2, 2),
        (128, None, 2, 3),
        (128, None, 3, 3),
        (128, 'bf16', 1, 1),
        (128, 'bf16', 1, 3),
        (128, 'bf16', 2, 2),
        (128, 'bf16', 2, 3),
        (128, 'bf16', 3, 3),
    ]

    for height, amp_dtype, bs, chunk in configs:
        o = deepcopy(base_opt)
        o.update(is_train=True, dist=False, rank=0, world_size=1, num_gpu=1)
        o['network_g']['train_frame_chunk_size'] = chunk
        o['train']['amp_dtype'] = amp_dtype
        o['train'].update(
            l_region_weight=2.5,
            f4_frame_weight=1.5,
            overlap_consistency_weight=0.2,
            grad_loss_weight=0.1,
            ssim_loss_weight=0.05,
        )
        prec = 'bf16' if amp_dtype else 'fp32'
        rec = {
            'height': height,
            'precision': prec,
            'batch_size_per_gpu': bs,
            'chunk': chunk,
            'status': 'ok',
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = None
        try:
            m = build_model(o)
            batch = {
                'lq_seq': torch.rand(bs, 9, 1, height, 512),
                'gt_seq': torch.rand(bs, 9, 1, height, 512),
                'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0).expand(bs, 9),
                'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0).expand(bs, 9, 4, 4, 1),
            }
            step_times = []
            for step in (1, 2, 3):
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
            sec_per_iter = float(np.mean(step_times[1:]))
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
            f"[train_bench] H={height:3d} {prec:4s} bs={bs} chunk={chunk} -> {rec['status']} "
            + (
                f"| {rec['sec_per_iter']:.3f}s/it | {rec['seqs_per_sec_2gpu_est']:.2f} seq/s(2GPU) "
                f"| alloc={rec['peak_allocated_gib']:.2f}G res={rec['peak_reserved_gib']:.2f}G"
                if rec['status'] == 'ok'
                else ''
            ),
            flush=True,
        )
        matrix.append(rec)
    return matrix


def benchmark_eval_bf16_and_batch() -> dict:
    """在 A_init (60k) 与 val.txt 前 10 条序列上验证 BF16 推理数值偏差及 eval_frame_batch_size 加速比。"""
    opt = yaml_load('options/SAR/cross_frame_v1/eval_a_init.yml')
    ds_opt = opt['datasets']['test']
    dataroot_lq = Path(ds_opt['dataroot_lq'])
    dataroot_gt = Path(ds_opt['dataroot_gt'])
    seq_names = [line.strip() for line in Path('docs/UCAN_exp1/splits/scene_split_v1/val.txt').read_text(encoding='utf-8').splitlines() if line.strip()][:10]

    ckpt = torch.load('experiments/ucan_a_init_scene_v1/models/net_g_best_psnr.pth', map_location='cpu', weights_only=True)
    state = ckpt['params']

    psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
    ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)

    # 先加载并预读取 10 条序列到内存，避免 I/O 干扰
    samples = []
    for s in seq_names:
        lq_u8 = [cv2.imread(str(dataroot_lq / s / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)]
        gt_u8 = [cv2.imread(str(dataroot_gt / s / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)]
        lq_t = torch.stack([img2tensor(np.ascontiguousarray(img[:, :, None]), bgr2rgb=False).div_(255.0) for img in lq_u8], dim=0).unsqueeze(0).cuda()
        samples.append((s, lq_t, gt_u8))

    fids = torch.arange(9, dtype=torch.long, device='cuda').unsqueeze(0)
    results = {}
    ref_outputs = None

    for prec in ('fp32', 'bf16'):
        for efbs in (1, 2, 3):
            key = f'{prec}_efbs{efbs}'
            net_opt = deepcopy(opt['network_g'])
            net_opt['eval_frame_batch_size'] = efbs
            net = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                net = build_network(net_opt).cuda().eval()
                net.load_backbone_state_dict(state, strict=True)
                use_bf16 = prec == 'bf16'

                # warmup
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    _ = net(samples[0][1], frame_ids=fids)
                torch.cuda.synchronize()

                psnrs, ssims, seq_times = [], [], []
                cur_outputs = []
                max_u8_diff = 0
                max_float_diff = 0.0

                with torch.no_grad():
                    for idx, (_, lq_t, gt_u8) in enumerate(samples):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                            out = net(lq_t, frame_ids=fids)
                        out = out.float()
                        torch.cuda.synchronize()
                        seq_times.append((time.perf_counter() - t0) * 1000.0)

                        pred_cpu = out[0].cpu()
                        cur_outputs.append(pred_cpu)
                        for t in range(9):
                            img_p = tensor2img([pred_cpu[t]])
                            psnrs.append(float(calculate_metric(dict(img=img_p, img2=gt_u8[t]), psnr_opt)))
                            ssims.append(float(calculate_metric(dict(img=img_p, img2=gt_u8[t]), ssim_opt)))
                            if ref_outputs is not None:
                                ref_u8 = tensor2img([ref_outputs[idx][t]])
                                max_u8_diff = max(max_u8_diff, int(np.max(np.abs(img_p.astype(np.int32) - ref_u8.astype(np.int32)))))
                        if ref_outputs is not None:
                            max_float_diff = max(max_float_diff, float((pred_cpu - ref_outputs[idx]).abs().max()))

                if ref_outputs is None and prec == 'fp32' and efbs == 1:
                    ref_outputs = cur_outputs

                mean_psnr = float(np.mean(psnrs))
                mean_ssim = float(np.mean(ssims))
                ref_psnr = results['fp32_efbs1']['mean_psnr'] if 'fp32_efbs1' in results else mean_psnr
                results[key] = {
                    'status': 'ok',
                    'precision': prec,
                    'eval_frame_batch_size': efbs,
                    'mean_psnr': mean_psnr,
                    'mean_ssim': mean_ssim,
                    'psnr_diff_vs_fp32_efbs1': float(mean_psnr - ref_psnr),
                    'max_float_diff_vs_fp32': float(max_float_diff),
                    'max_u8_diff_vs_fp32': int(max_u8_diff),
                    'seq_ms_mean': float(np.mean(seq_times)),
                    'peak_allocated_gib': float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
                    'peak_reserved_gib': float(torch.cuda.max_memory_reserved() / (1024 ** 3)),
                }
                print(
                    f"[eval_bench] {key:10s} | {results[key]['seq_ms_mean']:.1f} ms/seq "
                    f"| alloc={results[key]['peak_allocated_gib']:.2f}G res={results[key]['peak_reserved_gib']:.2f}G "
                    f"| psnr={mean_psnr:.6f} (diff={results[key]['psnr_diff_vs_fp32_efbs1']:+.6f}) "
                    f"| max_u8_diff={max_u8_diff}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                results[key] = {'status': 'OOM', 'precision': prec, 'eval_frame_batch_size': efbs}
                print(f"[eval_bench] {key:10s} -> OOM", flush=True)
            finally:
                if net is not None:
                    del net
                gc.collect()
                torch.cuda.empty_cache()
    return results


def main():
    assert torch.cuda.is_available(), 'CUDA device required'
    base_opt = yaml_load('options/SAR/v2_stabilize/C_v2a_stabilize_s42.yml')
    eval_res = benchmark_eval_bf16_and_batch()
    train_res = benchmark_train_matrix(base_opt)

    out_dir = Path('docs/UCAN_exp1/diagnostics/throughput_4090_v2')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'device_name': torch.cuda.get_device_name(0),
        'eval_10seqs_512x512': eval_res,
        'train_matrix': train_res,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved benchmark summary to {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
