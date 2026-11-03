"""UCAN 双 4090 有界短测：吞吐、整帧分片验证与原生保存恢复。"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.models import build_model
from basicsr.utils import get_root_logger, set_random_seed
from basicsr.utils.options import yaml_load


class RecordedSubset(Dataset):
    """记录各 rank 实际读取的验证索引，检查不等长及空分片。"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.opt = dict(dataset.opt, name=f'subset_{len(indices)}')
        self.seen = []

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        self.seen.append(index)
        return self.dataset[self.indices[index]]


def emit(stage, **values):
    if dist.get_rank() == 0:
        print(
            json.dumps(
                dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), stage=stage, **values),
                ensure_ascii=False),
            flush=True)


def synced(model):
    actual = torch.nn.utils.parameters_to_vector(model.net_g.parameters()).detach()
    expected = actual.clone()
    dist.broadcast(expected, src=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def step(model, batch, iteration):
    model.update_learning_rate(iteration, warmup_iter=-1)
    model.feed_data(batch)
    model.optimize_parameters(iteration)
    assert math.isfinite(float(model.log_dict['l_pix']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--burnin', type=int, default=10)
    parser.add_argument('--paper-adapted', action='store_true')
    parser.add_argument('--check-flow', action='store_true')
    parser.add_argument('--full-val', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.burnin < args.steps <= 200
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    set_random_seed(42 + rank)
    output = ROOT / 'experiments/ucan_ddp_tuning_20261103' / args.label
    output.mkdir(parents=True, exist_ok=True)
    get_root_logger(log_file=str(output / f'rank{rank}.log'), log_level=logging.WARNING)
    config = Path(args.config)
    opt = yaml_load(str(config))
    opt.update(is_train=True, dist=True, rank=rank, world_size=world, num_gpu=2)
    opt['datasets']['train']['batch_size_per_gpu'] = args.batch
    if args.paper_adapted:
        opt['train']['optim_g'].update(lr=3.75e-4, betas=[0.9, 0.99])
        opt['train']['total_iter'] = 400000
        opt['train']['scheduler']['milestones'] = [150000, 250000, 325000, 350000, 375000]
        opt['val']['val_freq'] = 10000
        opt['logger'].update(print_freq=134, save_checkpoint_freq=10000)
    for phase in ('train', 'val'):
        opt['datasets'][phase].update(phase=phase, scale=1)
    for key in ('models', 'training_states', 'visualization'):
        path = output / key
        path.mkdir(exist_ok=True)
        opt['path'][key] = str(path)
    if rank == 0:
        (output / 'runtime_options.json').write_text(json.dumps(opt, ensure_ascii=False, indent=2), encoding='utf-8')
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    model = build_model(deepcopy(opt))
    assert isinstance(model.net_g, DistributedDataParallel)
    assert all(value.device == torch.device(f'cuda:{rank}') for name, value in model.net_g.module.named_buffers()
               if name.endswith('inv_freq'))
    dataset = build_dataset(opt['datasets']['train'])
    sampler = EnlargedSampler(dataset, world, rank, 1)
    sampler.set_epoch(0)
    loader = build_dataloader(dataset, opt['datasets']['train'], num_gpu=2, dist=True, sampler=sampler, seed=42)
    iterator = iter(loader)
    records = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    emit('train_start', label=args.label, batch_per_gpu=args.batch, steps=args.steps)
    for iteration in range(1, args.steps + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = next(iterator)
        fetched = time.perf_counter()
        step(model, batch, iteration)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        records.append(
            dict(iter=iteration, seconds=elapsed, data_seconds=fetched - start, loss=float(model.log_dict['l_pix'])))
        if iteration == 1 or iteration == args.steps:
            for name, param in model.net_g.module.named_parameters():
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
        if iteration == 1 or iteration % 25 == 0 or iteration == args.steps:
            emit('training', label=args.label, **records[-1], peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    synced(model)
    measured = records[args.burnin:]
    local = dict(
        rank=rank,
        records=records,
        measured_seconds=sum(row['seconds'] for row in measured),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        (output / 'training_measurements.json').write_text(
            json.dumps(gathered, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.check_flow:
        model.save(0, args.steps)
        dist.barrier()
    del iterator, loader, dataset
    model.optimizer_g.zero_grad(set_to_none=True)
    for key in ('lq', 'gt', 'output'):
        if hasattr(model, key):
            delattr(model, key)
    gc.collect()
    validations = []
    if args.check_flow or args.full_val:
        dataset = build_dataset(opt['datasets']['val'])
        if args.check_flow:
            for indices in ([0, 4, 8], [4]):
                subset = RecordedSubset(dataset, indices)
                torch.cuda.empty_cache()
                model.validation(
                    DataLoader(subset, batch_size=1), current_iter=args.steps, tb_logger=None, save_img=False)
                assert subset.seen == list(range(rank, len(indices), world))
                assert model.net_g.training and model.net_g.module.training
                metrics = dict(model.metric_results)
                model.nondist_validation(
                    DataLoader(subset, batch_size=1), current_iter=args.steps, tb_logger=None, save_img=False)
                differences = {key: model.metric_results[key] - metrics[key] for key in metrics}
                print(
                    json.dumps(
                        dict(
                            stage='reference_diagnostic',
                            rank=rank,
                            size=len(indices),
                            distributed=metrics,
                            independent=model.metric_results,
                            difference=differences)),
                    flush=True)
                # 两卡 benchmark 可能选用不同卷积算法；严格对照单独固定后端，训练/完整验证仍用默认后端。
                need_control = torch.tensor(
                    int(any(abs(value) > 1e-6 for value in differences.values())), device='cuda')
                dist.all_reduce(need_control, op=dist.ReduceOp.MAX)
                controlled = bool(need_control.item())
                if controlled:
                    with torch.backends.cudnn.flags(benchmark=False, deterministic=True):
                        model.validation(
                            DataLoader(subset, batch_size=1), current_iter=args.steps, tb_logger=None, save_img=False)
                        metrics = dict(model.metric_results)
                        model.nondist_validation(
                            DataLoader(subset, batch_size=1), current_iter=args.steps, tb_logger=None, save_img=False)
                    emit(
                        'controlled_reference',
                        size=len(indices),
                        distributed=metrics,
                        independent=model.metric_results)
                assert all(math.isclose(metrics[k], model.metric_results[k], abs_tol=1e-6) for k in metrics)
                validations.append(
                    dict(
                        size=len(indices),
                        metrics=metrics,
                        reference_agrees=True,
                        controlled_backend=controlled,
                        default_reference_difference=differences))
        if args.full_val:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            start = time.perf_counter()
            model.validation(DataLoader(dataset, batch_size=1), current_iter=args.steps, tb_logger=None, save_img=False)
            torch.cuda.synchronize()
            dist.barrier()
            seconds = time.perf_counter() - start
            peak = [None] * world
            dist.all_gather_object(peak, torch.cuda.max_memory_allocated() / 2**30)
            assert all(math.isfinite(value) for value in model.metric_results.values())
            validations.append(
                dict(size=len(dataset), seconds=seconds, metrics=model.metric_results, peak_allocated_gib=peak))
            emit('full_validation', **validations[-1])
        step(model, batch, args.steps + 1)
        synced(model)
        emit('continued_after_validation', loss=float(model.log_dict['l_pix']))
    restored_ok = False
    if args.check_flow:
        saved_step = args.steps + 1
        model.save(0, saved_step)
        dist.barrier()
        state = torch.load(output / f'training_states/{saved_step}.state', map_location='cpu', weights_only=True)
        expected = {key: value.detach().cpu().clone() for key, value in model.net_g.module.state_dict().items()}
        # 先释放旧模型与训练激活，避免两个完整训练实例同时驻留。
        del model
        gc.collect()
        torch.cuda.empty_cache()
        opt['path']['pretrain_network_g'] = str(output / f'models/net_g_{saved_step}.pth')
        model = build_model(deepcopy(opt))
        model.resume_training(state)
        for key, value in model.net_g.module.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
        actual = model.optimizer_g.state_dict()
        assert actual['param_groups'] == state['optimizers'][0]['param_groups']
        for key, values in actual['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(value.cpu(), state['optimizers'][0]['state'][key][name], rtol=0, atol=0)
        assert model.schedulers[0].state_dict() == state['schedulers'][0]
        step(model, batch, saved_step + 1)
        synced(model)
        restored_ok = True
        emit('restored', iteration=saved_step + 1, loss=float(model.log_dict['l_pix']))
    measured_seconds = max(item['measured_seconds'] for item in gathered)
    result = dict(
        status='passed',
        label=args.label,
        args=vars(args),
        started=started,
        completed=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch=torch.__version__,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        seconds_per_step=measured_seconds / len(measured),
        samples_per_second=len(measured) * args.batch * world / measured_seconds,
        ranks=gathered,
        validations=validations,
        restore_checked=restored_ok)
    if rank == 0:
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    emit(
        'complete',
        label=args.label,
        samples_per_second=result['samples_per_second'],
        seconds_per_step=result['seconds_per_step'],
        peaks=[item['peak_allocated_gib'] for item in gathered])
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
