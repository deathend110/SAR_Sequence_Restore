"""冻结单帧 UCAN 权重下的 A 与 A+Fusion 重叠输出融合预备诊断脚本（指导书第 10.3 节）。"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.options import yaml_load


def load_sequences(split_file: Path, dataroot_lq: Path, dataroot_gt: Path):
    """按 split_file 加载 9 帧序列的路径列表。"""
    names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    sequences = []
    for name in names:
        lq_paths = [dataroot_lq / name / f'{t:03d}.png' for t in range(9)]
        gt_paths = [dataroot_gt / name / f'{t:03d}.png' for t in range(9)]
        for p in lq_paths + gt_paths:
            if not p.is_file():
                raise FileNotFoundError(f'Missing frame: {p}')
        sequences.append((name, lq_paths, gt_paths))
    return sequences


def overlap_consistency_mad(frames_uint8: np.ndarray) -> float:
    """计算同地面条带在不同帧之间的平均绝对差（uint8 0-255 尺度）。

    frames_uint8: [9, 512, 512], uint8/float64
    满足 t1 < t2 且 t1 + k1 == t2 + k2 的所有同地面 128px 条带对。
    """
    arr = frames_uint8.astype(np.float64)
    total_abs = 0.0
    total_count = 0
    for t1 in range(9):
        for k1 in range(4):
            s = t1 + k1
            p1 = arr[t1, :, k1 * 128:(k1 + 1) * 128]
            for t2 in range(t1 + 1, 9):
                k2 = s - t2
                if 0 <= k2 < 4:
                    p2 = arr[t2, :, k2 * 128:(k2 + 1) * 128]
                    total_abs += float(np.abs(p1 - p2).sum())
                    total_count += p1.size
    return total_abs / total_count if total_count > 0 else 0.0


def region_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray, c0: int, c1: int) -> tuple[float, float]:
    """按第 14.2 节口径计算水平子区域 [c0, c1) 的 PSNR（区域 MSE）和局部窗口 SSIM。"""
    p_sub = pred_u8[:, c0:c1]
    g_sub = gt_u8[:, c0:c1]
    psnr = float(calculate_metric(
        dict(img=p_sub, img2=g_sub),
        dict(type='calculate_psnr', crop_border=0, test_y_channel=False),
    ))
    ssim = float(calculate_metric(
        dict(img=p_sub, img2=g_sub),
        dict(type='calculate_ssim', crop_border=0, test_y_channel=False),
    ))
    return psnr, ssim


def overlap_fuse_float(pred_float: np.ndarray) -> np.ndarray:
    """对 9 帧浮点输出 [9, H, 512] 在 1536 画布上做同坐标等权平均再裁回 9 帧。"""
    _, height, width = pred_float.shape
    assert width == 512
    canvas_sum = np.zeros((height, 1536), dtype=np.float64)
    canvas_cnt = np.zeros((height, 1536), dtype=np.float64)
    for t in range(9):
        c0 = 128 * t
        canvas_sum[:, c0:c0 + 512] += pred_float[t].astype(np.float64)
        canvas_cnt[:, c0:c0 + 512] += 1.0
    fused = canvas_sum / canvas_cnt
    out = np.empty_like(pred_float, dtype=np.float64)
    for t in range(9):
        c0 = 128 * t
        out[t] = fused[:, c0:c0 + 512]
    return out


def float_to_uint8_frame(arr: np.ndarray) -> np.ndarray:
    """与 BasicSR tensor2img 完全一致的浮点 [0,1] -> uint8 转换。"""
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor2img([tensor])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', default='options/SAR/test_ucan_single.yml')
    parser.add_argument('--checkpoint', default='experiments/ucan_sar_single/models/net_g_300000.pth')
    parser.add_argument('--split-file', default='docs/splits/sar_single_seed42/val.txt')
    parser.add_argument('--out-dir', default='docs/UCAN_exp1/diagnostics/pre_diagnosis')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = yaml_load(args.opt)
    ds_opt = next(iter(opt['datasets'].values()))
    dataroot_lq = Path(ds_opt['dataroot_lq'])
    dataroot_gt = Path(ds_opt['dataroot_gt'])
    sequences = load_sequences(Path(args.split_file), dataroot_lq, dataroot_gt)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = build_network(opt['network_g']).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    param_key = opt['path'].get('param_key_g', 'params')
    state = ckpt[param_key] if param_key in ckpt else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
    ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)

    # 统计容器
    # frame_stats[method][t][metric] -> list
    methods = ('input', 'A', 'A_fusion')
    frame_stats = {m: {t: {'psnr': [], 'ssim': []} for t in range(9)} for m in methods}
    f4_strip_stats = {m: {k: {'psnr': [], 'ssim': []} for k in range(4)} for m in methods}
    f4_center_stats = {m: {'psnr': [], 'ssim': []} for m in methods}
    # 地面条带 s=4..7（覆盖所有同地面帧视角）
    ground_s_stats = {m: {s: {'psnr': [], 'ssim': []} for s in range(4, 8)} for m in methods}
    overlap_mads = {'input': [], 'GT': [], 'A': [], 'A_fusion': []}
    seq_times_a = []
    seq_times_af = []

    # GPU 预热
    if device.type == 'cuda':
        dummy = torch.zeros(1, 1, 512, 512, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = net(dummy)
        torch.cuda.synchronize()

    for seq_name, lq_paths, gt_paths in sequences:
        lq_u8 = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in lq_paths], axis=0)
        gt_u8 = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in gt_paths], axis=0)

        # 逐帧推理（保持严格 W=512 单帧独立前向与历史一致）
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_float = np.empty((9, 512, 512), dtype=np.float32)
        with torch.no_grad():
            for t in range(9):
                inp = img2tensor(np.ascontiguousarray(lq_u8[t, :, :, None]), bgr2rgb=False).div_(255.0).unsqueeze(0).to(device)
                out = net(inp)
                pred_float[t] = out[0, 0].detach().cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fused_float = overlap_fuse_float(pred_float)
        t2 = time.perf_counter()

        seq_times_a.append((t1 - t0) * 1000.0)
        seq_times_af.append((t2 - t0) * 1000.0)

        a_u8 = np.stack([float_to_uint8_frame(pred_float[t]) for t in range(9)], axis=0)
        af_u8 = np.stack([float_to_uint8_frame(fused_float[t]) for t in range(9)], axis=0)

        overlap_mads['input'].append(overlap_consistency_mad(lq_u8))
        overlap_mads['GT'].append(overlap_consistency_mad(gt_u8))
        overlap_mads['A'].append(overlap_consistency_mad(a_u8))
        overlap_mads['A_fusion'].append(overlap_consistency_mad(af_u8))

        pred_map = {'input': lq_u8, 'A': a_u8, 'A_fusion': af_u8}
        for m, imgs in pred_map.items():
            for t in range(9):
                frame_stats[m][t]['psnr'].append(float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), psnr_opt)))
                frame_stats[m][t]['ssim'].append(float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), ssim_opt)))
                for k in range(4):
                    s = t + k
                    if 4 <= s <= 7:
                        p, s_val = region_metrics(imgs[t], gt_u8[t], k * 128, (k + 1) * 128)
                        ground_s_stats[m][s]['psnr'].append(p)
                        ground_s_stats[m][s]['ssim'].append(s_val)

            # F4 四个条带与中心条带 [128, 384)
            for k in range(4):
                p, s_val = region_metrics(imgs[4], gt_u8[4], k * 128, (k + 1) * 128)
                f4_strip_stats[m][k]['psnr'].append(p)
                f4_strip_stats[m][k]['ssim'].append(s_val)
            pc, sc = region_metrics(imgs[4], gt_u8[4], 128, 384)
            f4_center_stats[m]['psnr'].append(pc)
            f4_center_stats[m]['ssim'].append(sc)

    # 写出逐帧位置 CSV
    frame_rows = []
    for t in range(9):
        row = {'frame_idx': t, 'count': len(sequences)}
        for m in methods:
            row[f'{m}_psnr_mean'] = float(np.mean(frame_stats[m][t]['psnr']))
            row[f'{m}_psnr_std'] = float(np.std(frame_stats[m][t]['psnr'], ddof=0))
            row[f'{m}_ssim_mean'] = float(np.mean(frame_stats[m][t]['ssim']))
            row[f'{m}_ssim_std'] = float(np.std(frame_stats[m][t]['ssim'], ddof=0))
        row['fusion_minus_A_psnr'] = row['A_fusion_psnr_mean'] - row['A_psnr_mean']
        row['fusion_minus_A_ssim'] = row['A_fusion_ssim_mean'] - row['A_ssim_mean']
        frame_rows.append(row)

    with (out_dir / 'frame_position_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    # 写出 F4 条带表
    f4_rows = []
    for k in range(4):
        row = {'region': f'F4_k{k}_s{4 + k}'}
        for m in methods:
            row[f'{m}_psnr'] = float(np.mean(f4_strip_stats[m][k]['psnr']))
            row[f'{m}_ssim'] = float(np.mean(f4_strip_stats[m][k]['ssim']))
        row['fusion_minus_A_psnr'] = row['A_fusion_psnr'] - row['A_psnr']
        row['fusion_minus_A_ssim'] = row['A_fusion_ssim'] - row['A_ssim']
        f4_rows.append(row)
    center_row = {'region': 'F4_center_k1_k2'}
    for m in methods:
        center_row[f'{m}_psnr'] = float(np.mean(f4_center_stats[m]['psnr']))
        center_row[f'{m}_ssim'] = float(np.mean(f4_center_stats[m]['ssim']))
    center_row['fusion_minus_A_psnr'] = center_row['A_fusion_psnr'] - center_row['A_psnr']
    center_row['fusion_minus_A_ssim'] = center_row['A_fusion_ssim'] - center_row['A_ssim']
    f4_rows.append(center_row)

    with (out_dir / 'f4_strip_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(f4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(f4_rows)

    # 写出地面条带 s=4..7 表
    gs_rows = []
    for s in range(4, 8):
        row = {'ground_strip': f's={s}'}
        for m in methods:
            row[f'{m}_psnr'] = float(np.mean(ground_s_stats[m][s]['psnr']))
            row[f'{m}_ssim'] = float(np.mean(ground_s_stats[m][s]['ssim']))
        row['fusion_minus_A_psnr'] = row['A_fusion_psnr'] - row['A_psnr']
        row['fusion_minus_A_ssim'] = row['A_fusion_ssim'] - row['A_ssim']
        gs_rows.append(row)
    with (out_dir / 'ground_strip_s4_7_comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(gs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gs_rows)

    summary = {
        'checkpoint': args.checkpoint,
        'split_file': args.split_file,
        'num_sequences': len(sequences),
        'num_frames': len(sequences) * 9,
        'overall': {
            m: {
                'psnr_mean': float(np.mean([v for t in range(9) for v in frame_stats[m][t]['psnr']])),
                'ssim_mean': float(np.mean([v for t in range(9) for v in frame_stats[m][t]['ssim']])),
            }
            for m in methods
        },
        'f4': {
            m: {
                'psnr_mean': float(np.mean(frame_stats[m][4]['psnr'])),
                'ssim_mean': float(np.mean(frame_stats[m][4]['ssim'])),
            }
            for m in methods
        },
        'f4_center_k1_k2': {
            m: {
                'psnr_mean': float(np.mean(f4_center_stats[m]['psnr'])),
                'ssim_mean': float(np.mean(f4_center_stats[m]['ssim'])),
            }
            for m in methods
        },
        'ground_s4_7_mean': {
            m: {
                'psnr_mean': float(np.mean([v for s in range(4, 8) for v in ground_s_stats[m][s]['psnr']])),
                'ssim_mean': float(np.mean([v for s in range(4, 8) for v in ground_s_stats[m][s]['ssim']])),
            }
            for m in methods
        },
        'overlap_consistency_mad': {k: float(np.mean(v)) for k, v in overlap_mads.items()},
        'latency_9frames_ms': {
            'A_p50': float(np.percentile(seq_times_a, 50)),
            'A_p95': float(np.percentile(seq_times_a, 95)),
            'A_fusion_p50': float(np.percentile(seq_times_af, 50)),
            'A_fusion_p95': float(np.percentile(seq_times_af, 95)),
        },
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
