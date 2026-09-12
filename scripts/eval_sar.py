"""使用官方 BasicSR 模型和指标，汇总 SAR 单帧、九帧位置及 sequence 结果。"""

import csv
from pathlib import Path

import numpy as np

from basicsr.data import build_dataloader, build_dataset
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import get_root_logger, tensor2img
from basicsr.utils.options import parse_options


METRIC_NAMES = (
    'input_psnr', 'output_psnr', 'delta_psnr',
    'input_ssim', 'output_ssim', 'delta_ssim',
)


def evaluate(model, test_loader, output_dir, logger):
    """逐帧计算，全部结束后写 CSV 并只打印一次总体指标；不写恢复图片。"""
    if len(test_loader.dataset) == 0:
        raise ValueError('SAR evaluation dataset is empty.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in test_loader:
        model.feed_data(batch)
        model.test()
        visuals = model.get_current_visuals()
        lq = tensor2img(visuals['lq'])
        restored = tensor2img(visuals['result'])
        gt = tensor2img(visuals['gt'])
        row = dict(sequence_name=batch['sequence_name'][0], frame_idx=int(batch['frame_idx'][0]))
        # 固定 uint8 image-based 协议，避免与浮点 tensor 指标混用。
        for metric in ('psnr', 'ssim'):
            metric_opt = dict(type=f'calculate_{metric}', crop_border=0, test_y_channel=False)
            input_value = calculate_metric(dict(img=lq, img2=gt), metric_opt)
            output_value = calculate_metric(dict(img=restored, img2=gt), metric_opt)
            row[f'input_{metric}'] = input_value
            row[f'output_{metric}'] = output_value
            row[f'delta_{metric}'] = output_value - input_value
        rows.append(row)

    rows.sort(key=lambda row: (row['sequence_name'], row['frame_idx']))
    with (output_dir / 'metrics_per_frame.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence_name', 'frame_idx', *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    frame_summary = []
    for frame_idx in range(9):
        frame_rows = [row for row in rows if row['frame_idx'] == frame_idx]
        summary = dict(frame_idx=frame_idx, count=len(frame_rows))
        for key in METRIC_NAMES:
            values = [row[key] for row in frame_rows]
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values, ddof=0))
        frame_summary.append(summary)
    with (output_dir / 'frame_position_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame_idx', 'count'] + [
            f'{key}_{stat}' for key in METRIC_NAMES for stat in ('mean', 'std')])
        writer.writeheader()
        writer.writerows(frame_summary)

    sequence_groups = {}
    for row in rows:
        sequence_groups.setdefault(row['sequence_name'], []).append(row)
    with (output_dir / 'metrics_per_sequence.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence_name', 'count'] + [
            f'{key}_mean' for key in METRIC_NAMES])
        writer.writeheader()
        for name, sequence_rows in sequence_groups.items():
            writer.writerow(dict(sequence_name=name, count=len(sequence_rows), **{
                f'{key}_mean': float(np.mean([row[key] for row in sequence_rows])) for key in METRIC_NAMES}))

    # 对全部帧等权求均值；先逐帧算 delta，再求平均。
    means = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_NAMES}
    logger.info('SAR summary | %s | %s', output_dir,
                ', '.join(f'mean {key}: {means[key]:.6f}' for key in METRIC_NAMES))
    return means


def main():
    root = Path(__file__).resolve().parents[1]
    opt, _ = parse_options(str(root), is_train=False)
    if opt['dist'] or opt['num_gpu'] not in (0, 1):
        raise ValueError('Independent SAR evaluation requires a single GPU (or CPU), without a launcher.')
    checkpoint = Path(opt['path']['pretrain_network_g'])
    result_root = Path(opt['path']['results_root']) / checkpoint.stem
    result_root.mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(log_file=str(result_root / 'eval.log'))
    for dataset_opt in opt['datasets'].values():
        dataset = build_dataset(dataset_opt)
        loader = build_dataloader(dataset, dataset_opt, num_gpu=opt['num_gpu'], dist=False, seed=opt['manual_seed'])
        # 每个数据集从新模型开始；HINT 网络内部 cache 的后续更新沿用原实现。
        model = build_model(opt)
        evaluate(model, loader, result_root / dataset_opt['name'], logger)


if __name__ == '__main__':
    main()
