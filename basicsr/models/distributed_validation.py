"""SAR 模型共用的分片验证：每张图仅处理一次，结束后归约全体指标。"""
import time
from os import path as osp

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img


class DistributedValidationMixin:

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        rank, world_size = dist.get_rank(), dist.get_world_size()
        metric_opts = self.opt['val'].get('metrics')
        metric_names = list(metric_opts) if metric_opts is not None else []
        # 不使用会补齐样本的 DistributedSampler，保证每张图只统计一次。
        local_loader = DataLoader(
            dataset, batch_size=1, sampler=range(rank, len(dataset), world_size),
            num_workers=0, collate_fn=dataloader.collate_fn)
        metric_sums = [0.0] * len(metric_names)
        count = 0
        start = time.perf_counter()
        logger = get_root_logger()
        if rank == 0:
            logger.info(f'Start distributed validation {dataset_name}: {len(dataset)} images, {world_size} ranks.')
        use_pbar = self.opt['val'].get('pbar', False) and rank == 0
        if use_pbar:
            pbar = tqdm(total=len(local_loader), unit='image', desc='Validation rank 0')

        for val_data in local_loader:
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data = {'img': sr_img}
            if 'gt' in visuals:
                metric_data['img2'] = tensor2img([visuals['gt']])
                del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    suffix = self.opt['val']['suffix'] or self.opt['name']
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{suffix}.png')
                imwrite(sr_img, save_img_path)
            for index, name in enumerate(metric_names):
                metric_sums[index] += calculate_metric(metric_data, metric_opts[name])
            count += 1
            if use_pbar:
                pbar.update(1)
        if use_pbar:
            pbar.close()

        # 连同实际样本数一次归约；空分片也参与，无指标时仍协调验证结束。
        totals = torch.tensor(metric_sums + [count], dtype=torch.float64, device=self.device)
        dist.all_reduce(totals)
        totals = totals.tolist()
        if metric_opts is not None:
            self.metric_results = {name: totals[index] / totals[-1] for index, name in enumerate(metric_names)}
            if rank == 0:
                self._initialize_best_metric_results(dataset_name)
                for name, value in self.metric_results.items():
                    self._update_best_metric_result(dataset_name, name, value, current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if rank == 0:
            logger.info(f'End distributed validation {dataset_name}: {int(totals[-1])} images, '
                        f'{time.perf_counter() - start:.2f} seconds.')
