"""由 torchrun 启动的 SwinIR 双卡回归检查，输出目录必须独立于正式实验。"""
import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from basicsr.data import build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.options import yaml_load


class RecordedSubset(Dataset):
    """记录真正读取的索引，检查分片是否重复、遗漏或读取了其他卡的数据。"""

    def __init__(self, dataset, size):
        self.dataset = dataset
        self.opt = dataset.opt
        self.size = size
        self.seen = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.seen.append(index)
        return self.dataset[index]


def assert_synced(model):
    # 比较全部参数，而非只比较 loss 或部分参数。
    actual = torch.nn.utils.parameters_to_vector(model.net_g.parameters()).detach()
    expected = actual.clone()
    dist.broadcast(expected, src=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def train_step(model, batch, step):
    model.update_learning_rate(step, warmup_iter=-1)
    model.feed_data(batch)
    model.optimize_parameters(step)
    assert math.isfinite(float(model.log_dict['l_pix']))
    assert_synced(model)


def check_validation(model, dataset, rank, output):
    for size in (3, 1):
        subset = RecordedSubset(dataset, size)
        loader = DataLoader(subset, batch_size=1)
        first_metrics = None
        for repeat in range(2):
            subset.seen.clear()
            model.validation(loader, current_iter=2, tb_logger=None, save_img=False)
            assert model.net_g.training and model.net_g.module.training
            assert subset.seen == list(range(rank, size, dist.get_world_size()))
            seen = [None] * dist.get_world_size()
            dist.all_gather_object(seen, subset.seen)
            assert sorted(index for indices in seen for index in indices) == list(range(size))
            distributed_metrics = dict(model.metric_results)
            if repeat == 0:
                first_metrics = distributed_metrics
            else:
                assert distributed_metrics == first_metrics
            if rank == 1:
                assert not hasattr(model, 'best_metric_results')

        # 两卡各自运行单卡参考，前向绕过 DDP，不产生通信或等待其他卡。
        model.nondist_validation(loader, current_iter=2, tb_logger=None, save_img=False)
        for name, value in distributed_metrics.items():
            assert math.isclose(value, model.metric_results[name], rel_tol=0, abs_tol=1e-6)
        if rank == 1:
            del model.best_metric_results

    assert not list((output / 'visualization').rglob('*.png'))
    # 无指标也必须完成集合通信；同时检查原有训练内图片保存路径。
    metric_opts = model.opt['val']['metrics']
    model.opt['val']['metrics'] = None
    model.validation(loader, current_iter=2, tb_logger=None, save_img=True)
    model.opt['val']['metrics'] = metric_opts
    if rank == 0:
        image_name = Path(dataset[0]['lq_path']).stem
        assert (output / 'visualization' / image_name / f'{image_name}_2.png').is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-state', type=Path, help='读取完整 training state，验证全部验证图像后再训练一步。')
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    set_random_seed(42 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    get_root_logger(log_file=str(args.output / f'rank{rank}.log'))
    opt = yaml_load('options/SAR/train_swinir_single.yml')
    opt.update(is_train=True, dist=True, rank=rank, world_size=dist.get_world_size())
    for key in ('models', 'training_states', 'visualization'):
        path = args.output / key
        path.mkdir(exist_ok=True)
        opt['path'][key] = str(path)
    for phase in ('train', 'val'):
        opt['datasets'][phase]['phase'] = phase
    if args.resume_state:
        state = torch.load(args.resume_state, map_location='cpu', weights_only=True)
        opt['path']['pretrain_network_g'] = str(
            args.resume_state.parent.parent / 'models' / f'net_g_{state["iter"]}.pth')
    else:
        opt['train']['scheduler']['milestones'] = [3]
    model = build_model(deepcopy(opt))
    assert isinstance(model.net_g, DistributedDataParallel)
    train = build_dataset(opt['datasets']['train'])
    batch = next(iter(DataLoader(train, batch_size=3, sampler=range(rank * 3, rank * 3 + 3))))
    val = build_dataset(opt['datasets']['val'])

    if args.resume_state:
        model.resume_training(state)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.validation(DataLoader(val, batch_size=1), current_iter=state['iter'], tb_logger=None, save_img=False)
        elapsed = time.perf_counter() - start
        peaks = [None] * dist.get_world_size()
        dist.all_gather_object(peaks, torch.cuda.max_memory_allocated() / 2**30)
        train_step(model, batch, state['iter'] + 1)
        report = dict(validation_frames=len(val), validation_seconds=elapsed,
                      frames_per_second=len(val) / elapsed, peak_allocated_gib=peaks,
                      metrics=model.metric_results, continued_iter=state['iter'] + 1,
                      loss=float(model.log_dict['l_pix']))
    else:
        for step in (1, 2):
            train_step(model, batch, step)
        model.save(0, 2)
        check_validation(model, val, rank, args.output)
        train_step(model, batch, 3)
        model.save(0, 3)
        # 保存仅在 rank 0 执行，测试需等文件写完后才能读取恢复状态。
        dist.barrier()
        state = torch.load(args.output / 'training_states/3.state', map_location='cpu', weights_only=True)
        expected = {key: value.cpu().clone() for key, value in model.net_g.module.state_dict().items()}
        del model
        gc.collect()
        torch.cuda.empty_cache()
        opt['path']['pretrain_network_g'] = str(args.output / 'models/net_g_3.pth')
        model = build_model(deepcopy(opt))
        model.resume_training(state)
        for key, value in model.net_g.module.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
        for optimizer, saved in zip(model.optimizers, state['optimizers']):
            actual = optimizer.state_dict()
            assert actual['param_groups'] == saved['param_groups']
            for param_id, values in actual['state'].items():
                for key, value in values.items():
                    torch.testing.assert_close(value.cpu(), saved['state'][param_id][key], rtol=0, atol=0)
        assert model.schedulers[0].state_dict() == state['schedulers'][0]
        train_step(model, batch, 4)
        assert math.isclose(model.optimizer_g.param_groups[0]['lr'], 1.732e-4 * 0.5)
        report = dict(validation_sizes=[3, 1], repeats=2, continued_iter=4, loss=float(model.log_dict['l_pix']))

    if rank == 0:
        (args.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('SAR_DDP_RESULT', json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
