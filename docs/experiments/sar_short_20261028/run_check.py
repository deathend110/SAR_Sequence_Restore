"""用现有 BasicSR 模型执行受步数限制的双卡 FP32 实验。"""

import argparse
import copy
import datetime
import hashlib
import json
import logging
import os
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import get_root_logger, set_random_seed, tensor2img

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/sar_short_20261028'
CHECKPOINTS = {
    'swinir': 'experiments/swinir_sar_single/models/net_g_493358.pth',
    'hint': 'experiments/hint_sar_single_archived_20261021_151605/models/net_g_813496.pth',
}


def emit(kind, **values):
    if dist.get_rank() == 0:
        record = dict(time=datetime.datetime.now().astimezone().isoformat(), kind=kind, **values)
        print(json.dumps(record, ensure_ascii=False), flush=True)


def write_json(path, value):
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_splits():
    """按场景选一条完整序列，保留所有 F0～F8，不混用验证与测试。"""
    for split in ('val', 'test'):
        names = (ROOT / f'docs/splits/sar_single_seed42/{split}.txt').read_text(encoding='utf-8').split()
        groups = {}
        for name in names:
            groups.setdefault(name.split('_rstart_')[0], []).append(name)
        rng = random.Random(42)
        selected = [rng.choice(sorted(groups[key])) for key in sorted(groups)]
        if dist.get_rank() == 0:
            (OUT / f'{split}_subset.txt').write_text('\n'.join(selected) + '\n', encoding='utf-8')
    dist.barrier()


def options(args):
    opt = yaml.safe_load((ROOT / f'options/SAR/train_{args.model}_single_q2.yml').read_text(encoding='utf-8'))
    opt.update(
        is_train=args.phase != 'baseline', dist=True, num_gpu=2, rank=dist.get_rank(), world_size=dist.get_world_size())
    opt['path']['pretrain_network_g'] = None if args.phase == 'warmup' else str(ROOT / CHECKPOINTS[args.model])
    opt['path']['resume_state'] = None
    opt['train']['total_iter'] = args.steps
    opt['train']['warmup_iter'] = args.warmup
    if args.phase == 'finetune':
        opt['train']['optim_g']['lr'] = 5e-5
        opt['train']['scheduler']['milestones'] = [args.steps * 3 // 4]
    else:
        # 启动对照和测速阶段不额外衰减，避免多个因素同时变化。
        opt['train']['scheduler']['milestones'] = []
    if args.batch:
        opt['datasets']['train']['batch_size_per_gpu'] = args.batch
    return opt


def dataset_options(opt, split):
    config = copy.deepcopy(opt['datasets']['val'])
    config.update(phase='val', scale=1, split_file=str(OUT / f'{split}_subset.txt'))
    return config


def evaluate(model, datasets, split, tag):
    """固定分片及遍历顺序，计算原评估脚本采用的 uint8 image-based 指标。"""
    rank, world = dist.get_rank(), dist.get_world_size()
    bare = model.get_bare_model(model.net_g)
    cached = getattr(bare, 'qv_cache', None)
    if hasattr(bare, 'qv_cache'):
        # 短程实验独立评估：每次冷缓存开始，结束恢复训练缓存，避免验证改变训练轨迹。
        bare.qv_cache = None
    loader = DataLoader(datasets[split], batch_size=1, sampler=range(rank, len(datasets[split]), world), num_workers=0)
    rows = []
    dist.barrier()
    started = time.perf_counter()
    try:
        for batch in loader:
            model.feed_data(batch)
            model.test()
            visuals = model.get_current_visuals()
            images = {name: tensor2img(visuals[name]) for name in ('lq', 'result', 'gt')}
            row = dict(sequence=batch['sequence_name'][0], frame=int(batch['frame_idx'][0]))
            for metric in ('psnr', 'ssim'):
                config = dict(type=f'calculate_{metric}', crop_border=0, test_y_channel=False)
                row[f'input_{metric}'] = calculate_metric(dict(img=images['lq'], img2=images['gt']), config)
                row[metric] = calculate_metric(dict(img=images['result'], img2=images['gt']), config)
            rows.append(row)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
    finally:
        if hasattr(bare, 'qv_cache'):
            bare.qv_cache = cached
    rows = sorted([row for part in gathered for row in part], key=lambda row: (row['sequence'], row['frame']))
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ('input_psnr', 'input_ssim', 'psnr', 'ssim')}
    summary.update(tag=tag, split=split, frames=len(rows), seconds=time.perf_counter() - started)
    summary['per_scene'] = {}
    for scene in sorted({row['sequence'].split('_rstart_')[0] for row in rows}):
        group = [row for row in rows if row['sequence'].split('_rstart_')[0] == scene]
        summary['per_scene'][scene] = {key: float(np.mean([row[key] for row in group])) for key in ('psnr', 'ssim')}
    summary['per_frame'] = {
        str(frame): {
            key: float(np.mean([row[key] for row in rows if row['frame'] == frame]))
            for key in ('psnr', 'ssim')
        }
        for frame in range(9)
    }
    write_json(OUT / 'metrics' / f'{tag}_{split}.json', dict(summary=summary, rows=rows))
    emit('evaluation', **{key: value for key, value in summary.items() if key not in ('per_scene', 'per_frame')})
    return summary


def training_loader(opt):
    config = copy.deepcopy(opt['datasets']['train'])
    config.update(phase='train', scale=1)
    dataset = build_dataset(config)
    sampler = EnlargedSampler(dataset, dist.get_world_size(), dist.get_rank(), 1)
    sampler.set_epoch(0)
    return build_dataloader(dataset, config, num_gpu=2, dist=True, sampler=sampler, seed=42)


def max_across_ranks(value):
    tensor = torch.tensor(float(value), device='cuda', dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for key, tensor in tensors.items():
        digest.update(key.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=CHECKPOINTS, required=True)
    parser.add_argument('--phase', choices=('baseline', 'finetune', 'batch', 'warmup'), required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=-1)
    parser.add_argument('--burnin', type=int, default=10)
    parser.add_argument('--label', help='独立复核时指定输出名称，避免覆盖首轮结果。')
    parser.add_argument('--audit-first', action='store_true', help='仅复核初始权重、首批输入和第一步 loss。')
    parser.add_argument('--deterministic', action='store_true', help='启动对照复核时固定后端并关闭 TF32。')
    args = parser.parse_args()
    assert args.steps <= 400, '短程验证最多运行 400 步。'
    assert args.phase == 'baseline' or args.steps > args.burnin >= 0
    assert args.phase not in ('finetune', 'warmup') or args.steps % 4 == 0
    os.chdir(ROOT)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    set_random_seed(42 + dist.get_rank())
    torch.backends.cudnn.benchmark = not args.deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    get_root_logger(log_level=logging.WARNING)
    prepare_splits()
    opt = options(args)
    tag = f'{args.phase}_{args.model}'
    if args.phase == 'batch':
        tag += f'_b{args.batch}'
    if args.phase == 'warmup':
        tag += f'_w{args.warmup}'
    if args.label:
        tag = args.label
    write_json(OUT / 'configs' / f'{tag}.json', opt)
    emit(
        'start',
        tag=tag,
        args=vars(args),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32)
    model = build_model(copy.deepcopy(opt))
    initial_digest = tensor_digest(model.get_bare_model(
        model.net_g).state_dict()) if args.audit_first or args.deterministic else None
    datasets = {}
    if args.phase != 'batch':
        datasets = {split: build_dataset(dataset_options(opt, split)) for split in ('val', 'test')}
    summaries = []
    if args.phase == 'baseline':
        for split in ('val', 'test'):
            summaries.append(evaluate(model, datasets, split, tag))
        write_json(OUT / f'{tag}.json', dict(evaluations=summaries))
        dist.destroy_process_group()
        return
    if args.phase == 'finetune':
        summaries.append(evaluate(model, datasets, 'val', tag + '_step0'))
    loader = training_loader(opt)
    iterator = iter(loader)
    records = []
    elapsed = []
    best = (-float('inf'), None)
    model_path = OUT / 'models' / f'{tag}_best.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == 'finetune':
        # 将训练前基线纳入选择，避免微调退化时仍强行选取微调权重。
        best = (summaries[0]['psnr'], 0)
        if dist.get_rank() == 0:
            torch.save(
                {
                    'params': {
                        key: value.detach().cpu()
                        for key, value in model.get_bare_model(model.net_g).state_dict().items()
                    }
                }, model_path)
    dist.barrier()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    peak_training = 0.0
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        batch = next(iterator)
        batch_digest = tensor_digest({
            'lq': batch['lq'],
            'gt': batch['gt']
        }) if args.audit_first or (args.deterministic and step == 1) else None
        data_seconds = time.perf_counter() - tick
        model.update_learning_rate(step, warmup_iter=args.warmup)
        model.feed_data(batch)
        model.optimize_parameters(step)
        torch.cuda.synchronize()
        duration = time.perf_counter() - tick
        elapsed.append(duration)
        loss = float(model.get_current_log()['l_pix'])
        if not np.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}: {loss}')
        record = dict(
            step=step, loss=loss, lr=model.get_current_learning_rate()[0], seconds=duration, data_seconds=data_seconds)
        records.append(record)
        if args.deterministic and step == 1:
            local = dict(rank=dist.get_rank(), initial_weights_sha256=initial_digest, first_batch_sha256=batch_digest)
            digests = [None] * dist.get_world_size()
            dist.all_gather_object(digests, local)
            write_json(OUT / f'{tag}_digests.json', digests)
        if args.audit_first:
            local = dict(
                rank=dist.get_rank(),
                initial_weights_sha256=initial_digest,
                first_batch_sha256=batch_digest,
                sequences=batch['sequence_name'],
                frames=batch['frame_idx'].tolist(),
                **record)
            audits = [None] * dist.get_world_size()
            dist.all_gather_object(audits, local)
            write_json(OUT / f'{tag}.json', dict(audits=audits))
            emit('audit', tag=tag, audits=audits)
            dist.destroy_process_group()
            return
        peak_training = max(peak_training, torch.cuda.max_memory_allocated() / 2**30)
        if step == 1 or step % 25 == 0 or step == args.steps:
            emit('train', tag=tag, **record)
        if args.phase in ('finetune', 'warmup') and step % (args.steps // 4) == 0:
            summary = evaluate(model, datasets, 'val', f'{tag}_step{step}')
            summaries.append(summary)
            if summary['psnr'] > best[0]:
                best = (summary['psnr'], step)
                if dist.get_rank() == 0:
                    bare = model.get_bare_model(model.net_g)
                    torch.save({'params': {
                        key: value.detach().cpu()
                        for key, value in bare.state_dict().items()
                    }}, model_path)
            torch.cuda.reset_peak_memory_stats()
    total_seconds = max_across_ranks(time.perf_counter() - started)
    peak = max_across_ranks(peak_training)
    # 稳态计时包含取数据、前后向、优化器及 DDP 同步，排除首批预热。
    measured = elapsed[args.burnin:]
    mean_seconds = max_across_ranks(statistics.mean(measured))
    median_seconds = max_across_ranks(statistics.median(measured))
    result = dict(
        tag=tag,
        steps=args.steps,
        batch_per_gpu=opt['datasets']['train']['batch_size_per_gpu'],
        world_size=2,
        mean_step_seconds=mean_seconds,
        median_step_seconds=median_seconds,
        samples_per_second=2 * opt['datasets']['train']['batch_size_per_gpu'] / mean_seconds,
        peak_allocated_gib=peak,
        wall_seconds=total_seconds,
        records=records,
        evaluations=summaries,
        best_psnr_step=best[1],
        checkpoint=str(model_path) if best[1] is not None else None)
    if args.phase == 'finetune':
        dist.barrier()
        model.get_bare_model(model.net_g).load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=True)['params'])
        result['test'] = evaluate(model, datasets, 'test', tag + '_selected')
    write_json(OUT / f'{tag}.json', result)
    emit('complete', **{key: value for key, value in result.items() if key not in ('records', 'evaluations', 'test')})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
