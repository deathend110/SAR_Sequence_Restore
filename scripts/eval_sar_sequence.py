"""统一跨帧 Token 序列评估与诊断脚本（第 14 节、第 13.3 节）。

支持：
  - A / B / C / D1 / D0 及第二轮各模型整帧（H=512）或纵向同步切片推理；
  - 输出与输出融合 X+Fusion 的全部九帧、F0-F8、F4 四条带 k=0..3、F4 中心条带 k=1,2、
    地面条带 s=4..7、H 内部与边界邻域、重叠区一致性 MAD；
  - 固定散射点 ROI 指标（峰值误差、位置偏移、弱点局部对比度、新增亮点）；
  - 可选诊断模式：帧排列一致性、错配身份、桥梁/两端屏蔽、16x9 注意力、d_i/c_j、detail 权重；
  - GPU 端到端与纯网络 P50/P95 时延及峰值显存统计。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs import build_network
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.metrics import calculate_metric
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.options import yaml_load


def overlap_fuse_float(pred_float: np.ndarray) -> np.ndarray:
    """将 [9, H, 512] 浮点输出按滑窗步长 128 在 1536 画布上做同坐标等权平均并裁回 9 帧。"""
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


def overlap_consistency_mad(frames_uint8: np.ndarray) -> float:
    """同地面条带在不同帧输出之间的平均绝对差（uint8 0-255 尺度）。"""
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


def float_to_uint8(arr: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor2img([tensor])


def region_psnr_ssim(pred_u8: np.ndarray, gt_u8: np.ndarray, c0: int, c1: int) -> tuple[float, float]:
    psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
    ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)
    p = float(calculate_metric(dict(img=pred_u8[:, c0:c1], img2=gt_u8[:, c0:c1]), psnr_opt))
    s = float(calculate_metric(dict(img=pred_u8[:, c0:c1], img2=gt_u8[:, c0:c1]), ssim_opt))
    return p, s


def build_fixed_scatter_rois(gt_dir: Path, split_file: Path, out_path: Path, top_k_per_seq: int = 3) -> list[dict]:
    """从测试集 F4 GT 确定性提取固定散射点 ROI（16x16 窗口，避开边界 8px）并持久化复用。"""
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding='utf-8'))

    names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    rois = []
    # 每隔一定步长抽样若干代表性序列，或覆盖全部序列的前 K 个局部强散射点
    for seq in names[::max(1, len(names) // 30)]:
        gt_f4 = cv2.imread(str(gt_dir / seq / '004.png'), cv2.IMREAD_GRAYSCALE).astype(np.float64)
        # 用 5x5 局部背景中值滤波差寻找强散射点
        bg = cv2.medianBlur(gt_f4.astype(np.uint8), 11).astype(np.float64)
        prom = gt_f4 - bg
        # 仅在中心与两侧取峰值点，避开边界 16px
        prom[:16, :] = -1e9
        prom[-16:, :] = -1e9
        prom[:, :16] = -1e9
        prom[:, -16:] = -1e9
        selected = []
        work = prom.copy()
        for _ in range(top_k_per_seq):
            idx = int(np.argmax(work))
            ry, rx = divmod(idx, 512)
            if work[ry, rx] <= 10.0:
                break
            selected.append((int(ry), int(rx)))
            work[max(0, ry - 20):min(512, ry + 21), max(0, rx - 20):min(512, rx + 21)] = -1e9
        if selected:
            rois.append({'sequence_name': seq, 'frame_idx': 4, 'centers': selected, 'half_win': 8})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rois, ensure_ascii=False, indent=2), encoding='utf-8')
    return rois


def compute_scatter_roi_metrics(pred_f4_u8: np.ndarray, gt_f4_u8: np.ndarray, centers: list[list[int]], half_win: int = 8):
    """计算固定散射点 ROI 的峰值误差、位置偏移、局部对比度误差与非目标区新增亮点。"""
    pred_f = pred_f4_u8.astype(np.float64)
    gt_f = gt_f4_u8.astype(np.float64)
    peak_errs, pos_shifts, contrast_errs, false_bright_ratios = [], [], [], []
    for ry, rx in centers:
        y0, y1 = ry - half_win, ry + half_win + 1
        x0, x1 = rx - half_win, rx + half_win + 1
        p_patch = pred_f[y0:y1, x0:x1]
        g_patch = gt_f[y0:y1, x0:x1]
        gy, gx = divmod(int(np.argmax(g_patch)), g_patch.shape[1])
        py, px = divmod(int(np.argmax(p_patch)), p_patch.shape[1])
        peak_errs.append(abs(float(p_patch.max() - g_patch.max())))
        pos_shifts.append(float(np.hypot(py - gy, px - gx)))
        # 局部对比度：峰值与周围环带均值之比（+1 防零）
        g_bg = float(np.median(g_patch)) + 1.0
        p_bg = float(np.median(p_patch)) + 1.0
        contrast_errs.append(abs((float(p_patch.max()) / p_bg) - (float(g_patch.max()) / g_bg)))
        # 新增亮点：ROI 内非峰邻域超过 GT 背景+40 且 GT 未达该水平的像素占比
        ring_mask = np.ones_like(g_patch, dtype=bool)
        ring_mask[max(0, gy - 2):min(g_patch.shape[0], gy + 3), max(0, gx - 2):min(g_patch.shape[1], gx + 3)] = False
        false_bright = (p_patch[ring_mask] > (g_patch[ring_mask] + 30.0)).mean()
        false_bright_ratios.append(float(false_bright))
    return peak_errs, pos_shifts, contrast_errs, false_bright_ratios


def infer_sequence_float(
    net,
    lq_seq: torch.Tensor,
    frame_ids: torch.Tensor,
    token_quality: torch.Tensor,
    slice_height: int | None = None,
    slice_stride: int | None = None,
    masked_source_frames: list[int] | None = None,
    return_telemetry: bool = False,
):
    """执行单条序列推理，返回 [9, H, 512] float32（可选同时返回整帧 telemetry）。"""
    if slice_height is None or slice_height >= lq_seq.shape[-2]:
        out = net(
            lq_seq,
            frame_ids=frame_ids,
            token_quality=token_quality,
            masked_source_frames=masked_source_frames,
            return_telemetry=return_telemetry,
        )
        if return_telemetry:
            pred, tele = out
            return pred[0, :, 0].detach().cpu().numpy(), tele
        return out[0, :, 0].detach().cpu().numpy(), {}

    stride = slice_stride or slice_height
    height = lq_seq.shape[-2]
    acc = torch.zeros_like(lq_seq, dtype=torch.float32)
    cnt = torch.zeros((1, 1, 1, height, 1), dtype=torch.float32, device=lq_seq.device)
    starts = list(range(0, height - slice_height + 1, stride))
    if starts[-1] + slice_height < height:
        starts.append(height - slice_height)
    for y0 in starts:
        y1 = y0 + slice_height
        pred_p = net(
            lq_seq[:, :, :, y0:y1, :],
            frame_ids=frame_ids,
            token_quality=token_quality,
            masked_source_frames=masked_source_frames,
            return_telemetry=False,
        )
        acc[:, :, :, y0:y1, :] += pred_p.float()
        cnt[:, :, :, y0:y1, :] += 1.0
    return (acc / cnt)[0, :, 0].detach().cpu().numpy(), {}


def evaluate_sequence_model(
    opt_path: Path,
    ckpt_path: Path,
    split_file: Path,
    out_dir: Path,
    slice_height: int | None = None,
    slice_stride: int | None = None,
    masked_source_frames: list[int] | None = None,
    roi_file: Path | None = None,
    save_telemetry_summary: bool = False,
    benchmark_latency: bool = True,
    save_img: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = yaml_load(str(opt_path))
    ds_opt = next(iter(opt['datasets'].values()))
    dataroot_lq = Path(ds_opt['dataroot_lq'])
    dataroot_gt = Path(ds_opt['dataroot_gt'])

    names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = build_network(opt['network_g']).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    param_key = opt['path'].get('param_key_g', 'params')
    state = ckpt[param_key] if (param_key is not None and param_key in ckpt) else ckpt
    net.load_state_dict(state, strict=True)
    net.eval()

    total_params = sum(p.numel() for p in net.parameters())
    psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
    ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)

    rois = build_fixed_scatter_rois(dataroot_gt, split_file, roi_file) if roi_file is not None else []
    roi_map = {item['sequence_name']: item for item in rois}

    base_frame_ids = torch.arange(9, dtype=torch.long, device=device).unsqueeze(0)
    base_q = compute_grid_quality_tensor(base_frame_ids[0], 4, 4).unsqueeze(0).to(device)

    variants = ('raw', 'fusion')
    frame_metrics = {v: {t: {'psnr': [], 'ssim': []} for t in range(9)} for v in variants}
    f4_strip = {v: {k: {'psnr': [], 'ssim': []} for k in range(4)} for v in variants}
    f4_center = {v: {'psnr': [], 'ssim': []} for v in variants}
    ground_s = {v: {s: {'psnr': [], 'ssim': []} for s in range(12)} for v in variants}
    h_interior = {v: {'psnr': [], 'ssim': []} for v in variants}
    h_boundary = {v: {'psnr': [], 'ssim': []} for v in variants}
    overlap_mads = {'raw': [], 'fusion': [], 'GT': [], 'LQ': []}
    seq_rows = []
    roi_stats = {v: {'peak_err': [], 'pos_shift': [], 'contrast_err': [], 'false_bright': []} for v in variants}

    # 聚合 F4 16x9 注意力热图（仅当启用 token_interaction 且 save_telemetry_summary）
    f4_attn_sum = None
    f4_attn_cnt = 0
    q_d_vals, q_c_vals, q_attn_recv = [], [], []
    detail_deltas = []

    with torch.no_grad():
        for seq_name in names:
            lq_u8 = np.stack([cv2.imread(str(dataroot_lq / seq_name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)])
            gt_u8 = np.stack([cv2.imread(str(dataroot_gt / seq_name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE) for t in range(9)])

            lq_t = torch.stack(
                [img2tensor(np.ascontiguousarray(lq_u8[t, :, :, None]), bgr2rgb=False).div_(255.0) for t in range(9)], dim=0
            ).unsqueeze(0).to(device)

            need_tele = save_telemetry_summary and (slice_height is None)
            pred_float, tele = infer_sequence_float(
                net,
                lq_t,
                base_frame_ids,
                base_q,
                slice_height=slice_height,
                slice_stride=slice_stride,
                masked_source_frames=masked_source_frames,
                return_telemetry=need_tele,
            )
            fused_float = overlap_fuse_float(pred_float)

            raw_u8 = np.stack([float_to_uint8(pred_float[t]) for t in range(9)])
            fus_u8 = np.stack([float_to_uint8(fused_float[t]) for t in range(9)])

            if save_img:
                for t in range(9):
                    imwrite_path = out_dir / 'restored' / seq_name / f'{t:03d}.png'
                    imwrite_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(imwrite_path), raw_u8[t])

            overlap_mads['raw'].append(overlap_consistency_mad(raw_u8))
            overlap_mads['fusion'].append(overlap_consistency_mad(fus_u8))
            overlap_mads['GT'].append(overlap_consistency_mad(gt_u8))
            overlap_mads['LQ'].append(overlap_consistency_mad(lq_u8))

            pred_dict = {'raw': raw_u8, 'fusion': fus_u8}
            seq_row = {'sequence_name': seq_name, 'scene': seq_name.split('_rstart_')[0]}
            for v, imgs in pred_dict.items():
                p_list, s_list = [], []
                for t in range(9):
                    p = float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), psnr_opt))
                    s = float(calculate_metric(dict(img=imgs[t], img2=gt_u8[t]), ssim_opt))
                    frame_metrics[v][t]['psnr'].append(p)
                    frame_metrics[v][t]['ssim'].append(s)
                    p_list.append(p)
                    s_list.append(s)
                    for k in range(4):
                        gs = t + k
                        gp, gss = region_psnr_ssim(imgs[t], gt_u8[t], k * 128, (k + 1) * 128)
                        ground_s[v][gs]['psnr'].append(gp)
                        ground_s[v][gs]['ssim'].append(gss)
                        if gs in (0, 1, 2, 9, 10, 11):
                            h_interior[v]['psnr'].append(gp)
                            h_interior[v]['ssim'].append(gss)
                        elif gs in (3, 8):
                            h_boundary[v]['psnr'].append(gp)
                            h_boundary[v]['ssim'].append(gss)
                for k in range(4):
                    fp, fss = region_psnr_ssim(imgs[4], gt_u8[4], k * 128, (k + 1) * 128)
                    f4_strip[v][k]['psnr'].append(fp)
                    f4_strip[v][k]['ssim'].append(fss)
                cp, cs = region_psnr_ssim(imgs[4], gt_u8[4], 128, 384)
                f4_center[v]['psnr'].append(cp)
                f4_center[v]['ssim'].append(cs)
                seq_row[f'{v}_psnr_mean'] = float(np.mean(p_list))
                seq_row[f'{v}_ssim_mean'] = float(np.mean(s_list))
                seq_row[f'{v}_f4_psnr'] = p_list[4]
                seq_row[f'{v}_f4_center_psnr'] = cp

                if seq_name in roi_map:
                    rc = roi_map[seq_name]
                    pe, ps_, ce, fb = compute_scatter_roi_metrics(imgs[4], gt_u8[4], rc['centers'], rc['half_win'])
                    roi_stats[v]['peak_err'].extend(pe)
                    roi_stats[v]['pos_shift'].extend(ps_)
                    roi_stats[v]['contrast_err'].extend(ce)
                    roi_stats[v]['false_bright'].extend(fb)
            seq_rows.append(seq_row)

            if need_tele and tele and 'attn_layers' in tele and tele['attn_layers']:
                # 取最后一层 F4 (token 64..79) 对 9 帧 (0..8) 汇总成 16x9
                last_attn = tele['attn_layers'][-1][0].mean(dim=0).cpu().numpy()  # [N, N]
                if last_attn.shape[0] == 144:
                    f4_to_all = last_attn[64:80, :].reshape(16, 9, 16).sum(axis=-1)  # [16, 9]
                    if f4_attn_sum is None:
                        f4_attn_sum = np.zeros_like(f4_to_all, dtype=np.float64)
                    f4_attn_sum += f4_to_all
                    f4_attn_cnt += 1
                if tele.get('quality_layers'):
                    ql = tele['quality_layers'][-1]
                    q_d_vals.append(ql['d_tgt'][0].mean(dim=-1).cpu().numpy())
                    q_c_vals.append(ql['c_src'][0].mean(dim=-1).cpu().numpy())
                    # 每个来源 j 被跨帧读取的平均权重
                    cross_mask = np.ones((144, 144), dtype=bool)
                    for ft in range(9):
                        cross_mask[ft * 16:(ft + 1) * 16, ft * 16:(ft + 1) * 16] = False
                    recv_col = (last_attn * cross_mask).sum(axis=0)
                    q_attn_recv.append(recv_col)
                if tele.get('detail_info'):
                    detail_deltas.append(tele['detail_info']['detail_delta_abs_mean'])

    # GPU 时延与峰值显存测量（第 14.4 节）
    latency_info = {}
    if benchmark_latency and device.type == 'cuda':
        dummy = torch.rand(1, 9, 1, 512, 512, device=device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(5):
                _ = infer_sequence_float(net, dummy, base_frame_ids, base_q, slice_height=slice_height, slice_stride=slice_stride)
        torch.cuda.synchronize()
        net_times = []
        e2e_times = []
        with torch.no_grad():
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred_f, _ = infer_sequence_float(net, dummy, base_frame_ids, base_q, slice_height=slice_height, slice_stride=slice_stride)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                _ = overlap_fuse_float(pred_f)
                t2 = time.perf_counter()
                net_times.append((t1 - t0) * 1000.0)
                e2e_times.append((t2 - t0) * 1000.0)
        latency_info = {
            'net_p50_ms': float(np.percentile(net_times, 50)),
            'net_p95_ms': float(np.percentile(net_times, 95)),
            'fusion_e2e_p50_ms': float(np.percentile(e2e_times, 50)),
            'fusion_e2e_p95_ms': float(np.percentile(e2e_times, 95)),
            'peak_allocated_gib': float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)),
            'peak_reserved_gib': float(torch.cuda.max_memory_reserved(device) / (1024 ** 3)),
        }

    # 写出 per-sequence CSV（用于按序列配对差值与 95% CI）
    with (out_dir / 'metrics_per_sequence.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seq_rows)

    # 写出 F0-F8 CSV
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
        'opt': str(opt_path),
        'checkpoint': str(ckpt_path),
        'split_file': str(split_file),
        'num_sequences': len(names),
        'num_frames': len(names) * 9,
        'total_params': total_params,
        'slice_height': slice_height,
        'slice_stride': slice_stride,
        'masked_source_frames': masked_source_frames,
        'variants': {},
        'overlap_consistency_mad': {k: float(np.mean(v)) for k, v in overlap_mads.items()},
        'latency_memory': latency_info,
    }
    for v in variants:
        summary['variants'][v] = {
            'nine_frame_psnr': float(np.mean([x for t in range(9) for x in frame_metrics[v][t]['psnr']])),
            'nine_frame_ssim': float(np.mean([x for t in range(9) for x in frame_metrics[v][t]['ssim']])),
            'f4_psnr': float(np.mean(frame_metrics[v][4]['psnr'])),
            'f4_ssim': float(np.mean(frame_metrics[v][4]['ssim'])),
            'f4_center_psnr': float(np.mean(f4_center[v]['psnr'])),
            'f4_center_ssim': float(np.mean(f4_center[v]['ssim'])),
            'f4_strips_psnr': {f'k{k}': float(np.mean(f4_strip[v][k]['psnr'])) for k in range(4)},
            'f4_strips_ssim': {f'k{k}': float(np.mean(f4_strip[v][k]['ssim'])) for k in range(4)},
            'ground_s4_7_psnr': float(np.mean([x for s in range(4, 8) for x in ground_s[v][s]['psnr']])),
            'ground_s4_7_ssim': float(np.mean([x for s in range(4, 8) for x in ground_s[v][s]['ssim']])),
            'ground_strips_psnr': {f's{s}': float(np.mean(ground_s[v][s]['psnr'])) for s in range(12)},
            'h_interior_psnr': float(np.mean(h_interior[v]['psnr'])),
            'h_interior_ssim': float(np.mean(h_interior[v]['ssim'])),
            'h_boundary_psnr': float(np.mean(h_boundary[v]['psnr'])),
            'h_boundary_ssim': float(np.mean(h_boundary[v]['ssim'])),
            'f0_f8_psnr': [float(np.mean(frame_metrics[v][t]['psnr'])) for t in range(9)],
            'f0_f8_ssim': [float(np.mean(frame_metrics[v][t]['ssim'])) for t in range(9)],
            'scatter_roi': {
                k: float(np.mean(vals)) if vals else None for k, vals in roi_stats[v].items()
            },
        }

    if f4_attn_sum is not None and f4_attn_cnt > 0:
        np.save(out_dir / 'f4_attn_16x9.npy', f4_attn_sum / f4_attn_cnt)
    if q_d_vals:
        np.savez(
            out_dir / 'quality_telemetry.npz',
            d_mean=np.mean(q_d_vals, axis=0),
            c_mean=np.mean(q_c_vals, axis=0),
            cross_recv_mean=np.mean(q_attn_recv, axis=0),
        )
    if detail_deltas:
        summary['detail_delta_abs_mean'] = float(np.mean(detail_deltas))

    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split-file', default='docs/UCAN_exp1/splits/scene_split_v1/test.txt')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--slice-height', type=int, default=None)
    parser.add_argument('--slice-stride', type=int, default=None)
    parser.add_argument('--mask-frames', type=int, nargs='*', default=None)
    parser.add_argument('--roi-file', default='docs/UCAN_exp1/splits/scene_split_v1/fixed_scatter_rois.json')
    parser.add_argument('--telemetry', action='store_true')
    parser.add_argument('--no-bench', action='store_true')
    parser.add_argument('--save-img', action='store_true')
    args = parser.parse_args()

    res = evaluate_sequence_model(
        opt_path=Path(args.opt),
        ckpt_path=Path(args.ckpt),
        split_file=Path(args.split_file),
        out_dir=Path(args.out_dir),
        slice_height=args.slice_height,
        slice_stride=args.slice_stride,
        masked_source_frames=args.mask_frames,
        roi_file=Path(args.roi_file) if args.roi_file else None,
        save_telemetry_summary=args.telemetry,
        benchmark_latency=not args.no_bench,
        save_img=args.save_img,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
