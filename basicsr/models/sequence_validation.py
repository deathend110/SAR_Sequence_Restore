"""九帧序列模型专用分布式验证：按完整序列分片、逐帧拆解计算 PSNR/SSIM 并汇总 F0-F8 与历史记录。"""

import json
import time
from os import path as osp
from pathlib import Path

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img


class SequenceDistributedValidationMixin:
    """按完整九帧序列分片的分布式验证（禁止拼图，每帧位置独立归约）。"""

    def _run_sequence_inference(self, net, lq_seq, frame_ids, token_quality):
        """支持整帧 H=512 与纵向同步切片（如 H=64, stride=32/64）两种推理模式。"""
        val_opt = self.opt.get('val', {})
        slice_h = val_opt.get('slice_eval_height')
        slice_stride = val_opt.get('slice_eval_stride') or slice_h

        if not slice_h or slice_h >= lq_seq.shape[-2]:
            return net(lq_seq, frame_ids=frame_ids, token_quality=token_quality)

        bsz, t_count, ch, height, width = lq_seq.shape
        acc = torch.zeros_like(lq_seq, dtype=torch.float32)
        cnt = torch.zeros((1, 1, 1, height, 1), dtype=torch.float32, device=lq_seq.device)

        starts = list(range(0, height - slice_h + 1, slice_stride))
        if starts[-1] + slice_h < height:
            starts.append(height - slice_h)

        for y0 in starts:
            y1 = y0 + slice_h
            pred_patch = net(lq_seq[:, :, :, y0:y1, :], frame_ids=frame_ids, token_quality=token_quality)
            acc[:, :, :, y0:y1, :] += pred_patch.float()
            cnt[:, :, :, y0:y1, :] += 1.0
        return acc / cnt

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        rank, world_size = dist.get_rank(), dist.get_world_size()
        self._run_sequence_validation(dataset, dataset_name, current_iter, tb_logger, save_img, rank, world_size, is_dist=True)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        self._run_sequence_validation(dataset, dataset_name, current_iter, tb_logger, save_img, rank=0, world_size=1, is_dist=False)

    def _run_sequence_validation(self, dataset, dataset_name, current_iter, tb_logger, save_img, rank, world_size, is_dist):
        logger = get_root_logger()
        exp_root = self.opt['path'].get('experiments_root')
        # BasicSR train.py 在结束全部 epoch 循环后常以 current_iter = total_iter + 1 或 +2 再次触发验证，
        # 若已在 total_iter 完成验证且无需保存图像，则直接跳过重复推理。
        train_opt = self.opt.get('train') or {}
        val_opt = self.opt.get('val') or {}
        total_iter = train_opt.get('total_iter')
        val_freq = val_opt.get('val_freq')
        val_iters = {int(v) for v in (val_opt.get('val_iters') or [])}
        already_validated_at_total = (
            total_iter is not None
            and (
                (int(total_iter) in val_iters)
                if val_iters
                else (bool(val_freq) and int(total_iter) % int(val_freq) == 0)
            )
        )
        if (
            self.opt.get('is_train')
            and not save_img
            and total_iter is not None
            and int(current_iter) > int(total_iter)
            and (
                already_validated_at_total
                or (
                    exp_root is not None
                    and (Path(exp_root) / 'val_history.json').is_file()
                )
            )
        ):
            if already_validated_at_total:
                if rank == 0:
                    logger.info(f'Skip duplicate end-of-training validation at iter {current_iter}.')
                return
            hist_path = Path(exp_root) / 'val_history.json'
            try:
                hist = json.loads(hist_path.read_text(encoding='utf-8'))
                if hist and int(hist[-1]['iter']) == int(total_iter):
                    if rank == 0:
                        logger.info(f'Skip duplicate end-of-training validation at iter {current_iter}.')
                    return
            except Exception:
                pass

        metric_opts = self.opt['val'].get('metrics')
        metric_names = list(metric_opts) if metric_opts is not None else []
        n_metrics = len(metric_names)

        # 不使用会补齐样本的 DistributedSampler，保证每条序列只验证一次。
        num_workers = 2 if (self.device.type == 'cuda' and len(dataset) >= 16) else 0
        local_loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=range(rank, len(dataset), world_size),
            num_workers=num_workers,
        )

        # 每个帧位置 t=0..8 分别累计指标和与计数
        frame_sums = torch.zeros((9, n_metrics), dtype=torch.float64, device=self.device)
        frame_counts = torch.zeros((9, 1), dtype=torch.float64, device=self.device)

        start = time.perf_counter()
        if rank == 0:
            logger.info(f'Start sequence validation {dataset_name}: {len(dataset)} sequences ({len(dataset)*9} frames), {world_size} ranks.')

        for seq_idx, val_data in enumerate(local_loader):
            seq_name = val_data['sequence_name'][0]
            self.feed_data(val_data)
            self.test()

            # 输出 [1, 9, 1, H, W] -> 严格逐帧拆解为 [1, H, W]，避免 tensor2img 拼图陷阱
            pred_seq = self.output[0].detach().cpu()  # [9, 1, H, W]
            gt_seq = self.gt[0].detach().cpu() if hasattr(self, 'gt') else None
            frame_ids = val_data['frame_ids'][0].tolist()

            for slot, t in enumerate(frame_ids):
                sr_img = tensor2img([pred_seq[slot]])
                metric_data = {'img': sr_img}
                if gt_seq is not None:
                    metric_data['img2'] = tensor2img([gt_seq[slot]])

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], seq_name, f'{t:03d}_{current_iter}.png'
                        )
                    else:
                        suffix = self.opt['val'].get('suffix') or self.opt['name']
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, seq_name, f'{t:03d}_{suffix}.png'
                        )
                    imwrite(sr_img, save_img_path)

                for m_idx, name in enumerate(metric_names):
                    frame_sums[t, m_idx] += float(calculate_metric(metric_data, metric_opts[name]))
                frame_counts[t, 0] += 1.0

            del self.lq, self.output
            if hasattr(self, 'gt'):
                del self.gt
            if self.device.type == 'cuda' and ((seq_idx + 1) % 8 == 0):
                torch.cuda.empty_cache()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        packed = torch.cat([frame_sums, frame_counts], dim=1)  # [9, n_metrics + 1]
        if is_dist:
            dist.all_reduce(packed)

        total_counts = packed[:, -1]
        total_frames = int(total_counts.sum().item())

        if metric_opts is not None:
            per_frame_metrics = {}
            overall = {}
            for m_idx, name in enumerate(metric_names):
                vals = []
                for t in range(9):
                    cnt_t = total_counts[t].item()
                    vals.append(packed[t, m_idx].item() / cnt_t if cnt_t > 0 else 0.0)
                per_frame_metrics[name] = vals
                overall[name] = float(packed[:, m_idx].sum().item() / max(total_frames, 1))

            self.metric_results = overall
            self.per_frame_metric_results = per_frame_metrics

            if rank == 0:
                self._initialize_best_metric_results(dataset_name)
                for name, value in self.metric_results.items():
                    self._update_best_metric_result(dataset_name, name, value, current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

                # 记录逐帧汇总与 val_history.json，便于选权与报告绘制
                f_str = ' | '.join(
                    f'F{t}:{per_frame_metrics[metric_names[0]][t]:.4f}' for t in range(9)
                )
                logger.info(f'Sequence per-frame {metric_names[0]} @ iter {current_iter}: {f_str}')

                exp_root = self.opt['path'].get('experiments_root')
                if exp_root is not None:
                    hist_path = Path(exp_root) / 'val_history.json'
                    history = []
                    if hist_path.is_file():
                        history = json.loads(hist_path.read_text(encoding='utf-8'))
                    history.append({
                        'iter': int(current_iter),
                        'dataset': dataset_name,
                        'total_frames': total_frames,
                        'metrics': overall,
                        'per_frame': per_frame_metrics,
                    })
                    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')

        if rank == 0:
            logger.info(
                f'End sequence validation {dataset_name}: {total_frames} frames, '
                f'{time.perf_counter() - start:.2f} seconds.'
            )
