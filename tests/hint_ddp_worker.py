"""HINT 完整结构双卡短流程；只使用训练和验证集，输出写入临时目录。"""
import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import get_root_logger, set_random_seed, tensor2img
from basicsr.utils.options import yaml_load
from swinir_ddp_worker import RecordedSubset, train_step


def check_validation(model, dataset, rank, size):
    subset = RecordedSubset(dataset, size)
    loader = DataLoader(subset, batch_size=1)
    net = model.net_g.module
    # 持久缓存会影响输出；参考计算必须从各 rank 相同的缓存快照和分片顺序开始。
    cache = net.qv_cache.clone()
    model.validation(loader, current_iter=2, tb_logger=None, save_img=False)
    distributed_metrics = dict(model.metric_results)
    expected_cache = net.qv_cache.clone()
    assert subset.seen == list(range(rank, size, dist.get_world_size()))
    assert model.net_g.training and net.training
    if rank == 1:
        assert not hasattr(model, 'best_metric_results')
    net.qv_cache = cache
    totals = [0., 0., 0.]
    for index in range(rank, size, dist.get_world_size()):
        batch = next(iter(DataLoader(RecordedSubset(dataset, index + 1), batch_size=1,
                                     sampler=[index])))
        model.feed_data(batch)
        model.test()
        images = model.get_current_visuals()
        data = dict(img=tensor2img(images['result']), img2=tensor2img(images['gt']))
        for i, metric in enumerate(('psnr', 'ssim')):
            totals[i] += calculate_metric(data, model.opt['val']['metrics'][metric])
        totals[2] += 1
    torch.testing.assert_close(net.qv_cache, expected_cache, rtol=0, atol=0)
    totals = torch.tensor(totals, dtype=torch.float64, device=model.device)
    dist.all_reduce(totals)
    assert int(totals[2]) == size
    for i, metric in enumerate(('psnr', 'ssim')):
        assert math.isclose(distributed_metrics[metric], float(totals[i] / totals[2]), abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    set_random_seed(42 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    get_root_logger(log_file=str(args.output / f'rank{rank}.log'))
    opt = yaml_load('options/SAR/train_hint_single.yml')
    opt.update(is_train=True, dist=True, rank=rank, world_size=dist.get_world_size())
    for key in ('models', 'training_states', 'visualization'):
        path = args.output / key
        path.mkdir(exist_ok=True)
        opt['path'][key] = str(path)
    for phase in ('train', 'val'):
        opt['datasets'][phase]['phase'] = phase
    opt['train']['scheduler']['milestones'] = [3]
    model = build_model(deepcopy(opt))
    assert model.net_g.find_unused_parameters
    train = build_dataset(opt['datasets']['train'])
    batch_size = opt['datasets']['train']['batch_size_per_gpu']
    batch = next(iter(DataLoader(train, batch_size=batch_size,
                                 sampler=range(rank * batch_size, (rank + 1) * batch_size))))
    for step in (1, 2):
        train_step(model, batch, step)
    val = build_dataset(opt['datasets']['val'])
    for size in (3, 1):
        check_validation(model, val, rank, size)
    train_step(model, batch, 3)
    model.save(0, 3)
    dist.barrier()
    state = torch.load(args.output / 'training_states/3.state', map_location='cpu', weights_only=True)
    expected = {key: value.cpu().clone() for key, value in model.net_g.module.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    opt['path']['pretrain_network_g'] = str(args.output / 'models/net_g_3.pth')
    model = build_model(deepcopy(opt))
    model.resume_training(state)
    assert model.net_g.module.qv_cache is None
    for key, value in model.net_g.module.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    assert model.schedulers[0].state_dict() == state['schedulers'][0]
    train_step(model, batch, 4)
    assert math.isclose(model.optimizer_g.param_groups[0]['lr'], opt['train']['optim_g']['lr'] * .5)
    if rank == 0:
        report = dict(validation_sizes=[3, 1], continued_iter=4, global_batch=batch_size * 2,
                      loss=float(model.log_dict['l_pix']))
        (args.output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('HINT_DDP_RESULT', json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
