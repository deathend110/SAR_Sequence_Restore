"""Step 1 零训练 512x1536 整画布推理上界诊断脚本。

复用 eval_sar_sequence.py 的 uint8 评估口径与分布式工具：
  - 将 9 帧 512x512 滑窗输入按步长 128 拼接为 512x1536 连续地面画布；
  - 使用冻结单帧 UCAN 骨干（如 A_init）在 512x1536 画布上直接推理，再按滑窗切回 9 帧；
  - 若单卡 24 GiB 显存在 512x1536 下 OOM，则自动安全回退至 slice_height=256, slice_stride=128，
    并支持 --mode frame512 在完全相同的切片参数下运行 512x512 逐帧公平对照。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basicsr.archs import build_network  # noqa: E402
from basicsr.utils import img2tensor  # noqa: E402
from basicsr.utils.options import yaml_load  # noqa: E402
from scripts.eval_sar_sequence import (  # noqa: E402
    _init_distributed_context,
    _load_eval_checkpoint,
    build_fixed_scatter_rois,
    compute_scatter_roi_metrics,
    float_to_uint8,
    overlap_consistency_mad,
    overlap_fuse_float,
    region_psnr_ssim,
)
from basicsr.metrics import calculate_metric  # noqa: E402


def stitch_frames_to_canvas(lq_seq: torch.Tensor, stride: int = 128) -> torch.Tensor:
    """将 [B, 9, 1, H, 512] 按滑窗步长 128 拼接为 [B, 1, H, 1536]（重叠区取平均）。"""
    bsz, t_count, ch, height, width = lq_seq.shape
    assert t_count == 9 and width == 512
    canvas_w = stride * (t_count - 1) + width
    canvas_sum = torch.zeros((bsz, ch, height, canvas_w), dtype=lq_seq.dtype, device=lq_seq.device)
    canvas_cnt = torch.zeros((1, 1, 1, canvas_w), dtype=lq_seq.dtype, device=lq_seq.device)
    for t in range(t_count):
        c0 = stride * t
        canvas_sum[:, :, :, c0:c0 + width] += lq_seq[:, t]
        canvas_cnt[:, :, :, c0:c0 + width] += 1.0
    return canvas_sum / canvas_cnt


def crop_canvas_to_frames(canvas: torch.Tensor, stride: int = 128, t_count: int = 9, width: int = 512) -> torch.Tensor:
    """将 [B, 1, H, 1536] 裁回 [B, 9, 1, H, 512]。"""
    frames = [canvas[:, :, :, stride * t:stride * t + width] for t in range(t_count)]
    return torch.stack(frames, dim=1)


def infer_backbone_2d(
    backbone: torch.nn.Module,
    img_in: torch.Tensor,
    slice_height: int | None = None,
    slice_stride: int | None = None,
) -> torch.Tensor:
    """对 [B, 1, H, W] 执行单帧骨干推理，支持全高或纵向滑窗切片。"""
    if slice_height is None or slice_height >= img_in.shape[-2]:
        return backbone(img_in)

    stride = slice_stride or slice_height
    height = img_in.shape[-2]
    acc = torch.zeros_like(img_in, dtype=torch.float32)
    cnt = torch.zeros((1, 1, height, 1), dtype=torch.float32, device=img_in.device)
    starts = list(range(0, height - slice_height + 1, stride))
    if starts[-1] + slice_height < height:
        starts.append(height - slice_height)
    for y0 in starts:
        y1 = y0 + slice_height
        acc[:, :, y0:y1, :] += backbone(img_in[:, :, y0:y1, :]).float()
        cnt[:, :, y0:y1, :] += 1.0
    return acc / cnt


def probe_slice_fallback(
    backbone: torch.nn.Module,
    device: torch.device,
    width: int,
    slice_height: int | None,
    slice_stride: int | None,
    is_dist: bool,
) -> tuple[int | None, int | None, bool]:
    """在 24 GiB 显存上探针测试全高推理是否 OOM，若 OOM 则自动回退至 slice 256/128。"""
    if slice_height is not None or device.type != 'cuda':
        return slice_height, slice_stride, False

    oom_flag = 0
    try:
        torch.cuda.empty_cache()
        dummy = torch.rand(1, 1, 512, width, device=device)
        with torch.no_grad():
            _ = backbone(dummy)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        oom_flag = 1
    finally:
        if oom_flag:
            torch.cuda.empty_cache()

    if is_dist:
        flag_t = torch.tensor([oom_flag], dtype=torch.int32, device=device)
        dist.all_reduce(flag_t, op=dist.ReduceOp.MAX)
        oom_flag = int(flag_t.item())

    if oom_flag:
        return 256, 128, True
    return None, None, False


def run_diagnosis(
    opt_path: Path,
    ckpt_path: Path,
    split_file: Path,
    out_dir: Path,
    mode: str = 'canvas1536',
    slice_height: int | None = None,
    slice_stride: int | None = None,
    roi_file: Path | None = None,
    benchmark_latency: bool = True,
):
    if mode not in ('canvas1536', 'frame512'):
        raise ValueError(f'Unsupported mode: {mode}')

    is_dist, rank, world_size, device, created_pg = _init_distributed_context()
    try:
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
        opt = yaml_load(str(opt_path))
        ds_opt = next(iter(opt['datasets'].values()))
        dataroot_lq = Path(ds_opt['dataroot_lq'])
        dataroot_gt = Path(ds_opt['dataroot_gt'])

        names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        net = build_network(opt['network_g']).to(device)
        param_key = opt['path'].get('param_key_g', 'params')
        _load_eval_checkpoint(net, ckpt_path, param_key)
        net.eval()
        backbone = net.backbone
        total_params = sum(p.numel() for p in backbone.parameters())

        probe_w = 1536 if mode == 'canvas1536' else 512
        eff_sh, eff_ss, oom_fallback = probe_slice_fallback(
            backbone, device, probe_w, slice_height, slice_stride, is_dist
        )
        if rank == 0:
            print(
                f'[diagnose_canvas_1536] mode={mode}, split={split_file.name}, seqs={len(names)}, '
                f'slice_height={eff_sh}, slice_stride={eff_ss}, oom_fallback={oom_fallback}',
                flush=True,
            )

        if roi_file is not None:
            if rank == 0:
                rois = build_fixed_scatter_rois(dataroot_gt, split_file, roi_file)
            if is_dist:
                dist.barrier()
            if rank != 0:
                rois = build_fixed_scatter_rois(dataroot_gt, split_file, roi_file)
        else:
            rois = []
        roi_map = {item['sequence_name']: item for item in rois}

        psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
        ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)
        variants = ('raw', 'fusion') if mode == 'frame512' else ('raw',)

        local_indices = list(range(rank, len(names), world_size))
        local_records = []

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            for seq_idx in local_indices:
                seq_name = names[seq_idx]
                lq_u8 = np.stack([cv2.imread(str(dataroot_lq / seq_name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)])
                gt_u8 = np.stack([cv2.imread(str(dataroot_gt / seq_name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)])

                lq_t = torch.stack(
                    [img2tensor(np.ascontiguousarray(lq_u8[t, :, :, None]), bgr2rgb=False).div_(255.0) for t in range(9)],
                    dim=0,
                ).unsqueeze(0).to(device)  # [1, 9, 1, 512, 512]

                if mode == 'canvas1536':
                    canvas_in = stitch_frames_to_canvas(lq_t, stride=128)  # [1, 1, 512, 1536]
                    canvas_out = infer_backbone_2d(backbone, canvas_in, slice_height=eff_sh, slice_stride=eff_ss)
                    pred_seq = crop_canvas_to_frames(canvas_out, stride=128)
                    pred_float = pred_seq[0, :, 0].detach().cpu().numpy()
                else:
                    pred_list = []
                    for t in range(9):
                        out_t = infer_backbone_2d(backbone, lq_t[:, t], slice_height=eff_sh, slice_stride=eff_ss)
                        pred_list.append(out_t[0, 0].detach().cpu().numpy())
                    pred_float = np.stack(pred_list, axis=0)

                raw_u8 = np.stack([float_to_uint8(pred_float[t]) for t in range(9)])
                pred_dict = {'raw': raw_u8}
                seq_mads = {
                    'raw': overlap_consistency_mad(raw_u8),
                    'GT': overlap_consistency_mad(gt_u8),
                    'LQ': overlap_consistency_mad(lq_u8),
                }
                if mode == 'frame512':
                    fused_float = overlap_fuse_float(pred_float)
                    fus_u8 = np.stack([float_to_uint8(fused_float[t]) for t in range(9)])
                    pred_dict['fusion'] = fus_u8
                    seq_mads['fusion'] = overlap_consistency_mad(fus_u8)

                seq_row = {'sequence_name': seq_name, 'scene': seq_name.split('_rstart_')[0]}
                seq_metrics_per_v = {}
                for v, imgs in pred_dict.items():
                    p_list, s_list = [], []
                    ground_entries = []
                    for t in range(9):
                        p = float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), psnr_opt))
                        s = float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), ssim_opt))
                        p_list.append(p)
                        s_list.append(s)
                        for k in range(4):
                            gs = t + k
                            gp, gss = region_psnr_ssim(imgs[t], gt_u8[t], k * 128, (k + 1) * 128)
                            ground_entries.append((gs, gp, gss))
                    f4_strips_p, f4_strips_s = [], []
                    for k in range(4):
                        fp, fss = region_psnr_ssim(imgs[4], gt_u8[4], k * 128, (k + 1) * 128)
                        f4_strips_p.append(fp)
                        f4_strips_s.append(fss)
                    cp, cs = region_psnr_ssim(imgs[4], gt_u8[4], 128, 384)
                    seq_row[f'{v}_psnr_mean'] = float(np.mean(p_list))
                    seq_row[f'{v}_ssim_mean'] = float(np.mean(s_list))
                    seq_row[f'{v}_f4_psnr'] = p_list[4]
                    seq_row[f'{v}_f4_center_psnr'] = cp

                    roi_tuple = None
                    if seq_name in roi_map:
                        rc = roi_map[seq_name]
                        roi_tuple = compute_scatter_roi_metrics(imgs[4], gt_u8[4], rc['centers'], rc['half_win'])

                    seq_metrics_per_v[v] = {
                        'p_list': p_list,
                        's_list': s_list,
                        'ground_entries': ground_entries,
                        'f4_strips_p': f4_strips_p,
                        'f4_strips_s': f4_strips_s,
                        'f4_center': (cp, cs),
                        'roi_tuple': roi_tuple,
                    }

                local_records.append((seq_idx, {
                    'seq_row': seq_row,
                    'seq_mads': seq_mads,
                    'per_v': seq_metrics_per_v,
                }))

        eval_peak_alloc = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == 'cuda' else 0.0
        eval_peak_res = float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)) if device.type == 'cuda' else 0.0

        local_payload = {
            'records': local_records,
            'eval_peak_alloc_gib': eval_peak_alloc,
            'eval_peak_res_gib': eval_peak_res,
        }
        if is_dist:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_payload)
        else:
            gathered = [local_payload]

        if is_dist:
            dist.barrier()

        latency_info = {}
        if rank == 0 and benchmark_latency and device.type == 'cuda':
            dummy_seq = torch.rand(1, 9, 1, 512, 512, device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(3):
                    if mode == 'canvas1536':
                        _ = crop_canvas_to_frames(
                            infer_backbone_2d(backbone, stitch_frames_to_canvas(dummy_seq), eff_sh, eff_ss)
                        )
                    else:
                        for t in range(9):
                            _ = infer_backbone_2d(backbone, dummy_seq[:, t], eff_sh, eff_ss)
            torch.cuda.synchronize(device)
            net_times = []
            with torch.no_grad():
                for _ in range(10):
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    if mode == 'canvas1536':
                        _ = crop_canvas_to_frames(
                            infer_backbone_2d(backbone, stitch_frames_to_canvas(dummy_seq), eff_sh, eff_ss)
                        )
                    else:
                        for t in range(9):
                            _ = infer_backbone_2d(backbone, dummy_seq[:, t], eff_sh, eff_ss)
                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    net_times.append((t1 - t0) * 1000.0)
            latency_info = {
                'net_p50_ms': float(np.percentile(net_times, 50)),
                'net_p95_ms': float(np.percentile(net_times, 95)),
                'peak_allocated_gib': float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
                'peak_reserved_gib': float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
            }

        if is_dist:
            dist.barrier()
        if rank != 0:
            return {}

        all_records = []
        for payload in gathered:
            all_records.extend(payload['records'])
        all_records.sort(key=lambda item: item[0])

        frame_metrics = {v: {t: {'psnr': [], 'ssim': []} for t in range(9)} for v in variants}
        f4_strip = {v: {k: {'psnr': [], 'ssim': []} for k in range(4)} for v in variants}
        f4_center = {v: {'psnr': [], 'ssim': []} for v in variants}
        ground_s = {v: {s: {'psnr': [], 'ssim': []} for s in range(12)} for v in variants}
        h_interior = {v: {'psnr': [], 'ssim': []} for v in variants}
        h_boundary = {v: {'psnr': [], 'ssim': []} for v in variants}
        overlap_mads = {k: [] for k in all_records[0][1]['seq_mads']}
        seq_rows = []
        roi_stats = {v: {'peak_err': [], 'pos_shift': [], 'contrast_err': [], 'false_bright': []} for v in variants}

        for _, rec in all_records:
            seq_rows.append(rec['seq_row'])
            for mk, mv in rec['seq_mads'].items():
                overlap_mads[mk].append(mv)
            for v in variants:
                vm = rec['per_v'][v]
                for t in range(9):
                    frame_metrics[v][t]['psnr'].append(vm['p_list'][t])
                    frame_metrics[v][t]['ssim'].append(vm['s_list'][t])
                for gs, gp, gss in vm['ground_entries']:
                    ground_s[v][gs]['psnr'].append(gp)
                    ground_s[v][gs]['ssim'].append(gss)
                    if gs in (0, 1, 2, 9, 10, 11):
                        h_interior[v]['psnr'].append(gp)
                        h_interior[v]['ssim'].append(gss)
                    elif gs in (3, 8):
                        h_boundary[v]['psnr'].append(gp)
                        h_boundary[v]['ssim'].append(gss)
                for k in range(4):
                    f4_strip[v][k]['psnr'].append(vm['f4_strips_p'][k])
                    f4_strip[v][k]['ssim'].append(vm['f4_strips_s'][k])
                cp, cs = vm['f4_center']
                f4_center[v]['psnr'].append(cp)
                f4_center[v]['ssim'].append(cs)
                if vm['roi_tuple'] is not None:
                    pe, ps_, ce, fb = vm['roi_tuple']
                    roi_stats[v]['peak_err'].extend(pe)
                    roi_stats[v]['pos_shift'].extend(ps_)
                    roi_stats[v]['contrast_err'].extend(ce)
                    roi_stats[v]['false_bright'].extend(fb)

        with (out_dir / 'metrics_per_sequence.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            writer.writeheader()
            writer.writerows(seq_rows)

        fpos_rows = []
        for t in range(9):
            row = {'frame_idx': t, 'count': len(names)}
            for v in variants:
                row[f'{v}_psnr_mean'] = float(np.mean(frame_metrics[v][t]['psnr']))
                row[f'{v}_psnr_std'] = float(np.std(frame_metrics[v][t]['psnr'], ddof=0))
                row[f'{v}_ssim_mean'] = float(np.mean(frame_metrics[v][t]['ssim']))
                row[f'{v}_ssim_std'] = float(np.std(frame_metrics[v][t]['ssim'], ddof=0))
            fpos_rows.append(row)
        with (out_dir / 'frame_position_summary.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(fpos_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fpos_rows)

        summary = {
            'mode': mode,
            'opt': str(opt_path),
            'checkpoint': str(ckpt_path),
            'split_file': str(split_file),
            'num_sequences': len(names),
            'num_frames': len(names) * 9,
            'total_params': total_params,
            'slice_height': eff_sh,
            'slice_stride': eff_ss,
            'oom_fallback': oom_fallback,
            'variants': {},
            'overlap_consistency_mad': {k: float(np.mean(v)) for k, v in overlap_mads.items()},
            'latency_memory': latency_info,
        }
        for v in variants:
            f0_f8_p = [float(np.mean(frame_metrics[v][t]['psnr'])) for t in range(9)]
            v_valley_depth = 0.5 * (f0_f8_p[0] + f0_f8_p[8]) - f0_f8_p[4]
            summary['variants'][v] = {
                'nine_frame_psnr': float(np.mean([x for t in range(9) for x in frame_metrics[v][t]['psnr']])),
                'nine_frame_ssim': float(np.mean([x for t in range(9) for x in frame_metrics[v][t]['ssim']])),
                'f4_psnr': float(np.mean(frame_metrics[v][4]['psnr'])),
                'f4_ssim': float(np.mean(frame_metrics[v][4]['ssim'])),
                'f4_center_psnr': float(np.mean(f4_center[v]['psnr'])),
                'f4_center_ssim': float(np.mean(f4_center[v]['ssim'])),
                'v_valley_depth': float(v_valley_depth),
                'f4_strips_psnr': {f'k{k}': float(np.mean(f4_strip[v][k]['psnr'])) for k in range(4)},
                'f4_strips_ssim': {f'k{k}': float(np.mean(f4_strip[v][k]['ssim'])) for k in range(4)},
                'ground_s4_7_psnr': float(np.mean([x for s in range(4, 8) for x in ground_s[v][s]['psnr']])),
                'ground_s4_7_ssim': float(np.mean([x for s in range(4, 8) for x in ground_s[v][s]['ssim']])),
                'ground_strips_psnr': {f's{s}': float(np.mean(ground_s[v][s]['psnr'])) for s in range(12)},
                'h_interior_psnr': float(np.mean(h_interior[v]['psnr'])),
                'h_interior_ssim': float(np.mean(h_interior[v]['ssim'])),
                'h_boundary_psnr': float(np.mean(h_boundary[v]['psnr'])),
                'h_boundary_ssim': float(np.mean(h_boundary[v]['ssim'])),
                'f0_f8_psnr': f0_f8_p,
                'f0_f8_ssim': [float(np.mean(frame_metrics[v][t]['ssim'])) for t in range(9)],
                'scatter_roi': {
                    k: float(np.mean(vals)) if vals else None for k, vals in roi_stats[v].items()
                },
            }

        (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', default='options/SAR/cross_frame_v1/eval_a_init.yml')
    parser.add_argument('--ckpt', default='experiments/ucan_a_init_scene_v1/models/net_g_best_psnr.pth')
    parser.add_argument('--split-file', default='docs/UCAN_exp1/splits/scene_split_v1/test.txt')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--mode', choices=('canvas1536', 'frame512'), default='canvas1536')
    parser.add_argument('--slice-height', type=int, default=None)
    parser.add_argument('--slice-stride', type=int, default=None)
    parser.add_argument('--roi-file', default='')
    parser.add_argument('--no-bench', action='store_true')
    args = parser.parse_args()

    res = run_diagnosis(
        opt_path=Path(args.opt),
        ckpt_path=Path(args.ckpt),
        split_file=Path(args.split_file),
        out_dir=Path(args.out_dir),
        mode=args.mode,
        slice_height=args.slice_height,
        slice_stride=args.slice_stride,
        roi_file=Path(args.roi_file) if args.roi_file else None,
        benchmark_latency=not args.no_bench,
    )
    if res:
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
