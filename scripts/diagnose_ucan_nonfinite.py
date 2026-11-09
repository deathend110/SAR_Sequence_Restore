"""从检查点续跑 UCAN，保存异常现场并逐算子定位首个非有限张量。"""
import argparse
import json
import os
from pathlib import Path
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from basicsr.archs import build_network
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.losses import build_loss
from basicsr.utils import set_random_seed
from basicsr.utils.options import yaml_load


def tensors(value):
    return [v for v in tree_flatten(value)[0] if isinstance(v, torch.Tensor)]


def summary(value):
    result = []
    for t in tensors(value):
        t = t.detach()
        item = dict(shape=list(t.shape), dtype=str(t.dtype))
        if t.is_floating_point() and t.numel():
            finite = torch.isfinite(t)
            item.update(nan=int(t.isnan().sum()), inf=int(t.isinf().sum()))
            good = t[finite]
            item['finite_abs_max'] = float(good.abs().max()) if good.numel() else None
        result.append(item)
    return result


class FirstNonfinite(TorchDispatchMode):
    """仅重放失败 batch 时启用；逐算子同步，避免正常续跑承担巨大开销。"""

    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.report = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        # empty 系列尚未写入数据，不能把未初始化的显存当成计算异常。
        if str(func).split('.')[1] in ('empty', 'empty_like', 'empty_strided', 'new_empty', 'new_empty_strided'):
            return out
        for t in tensors(out):
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                self.report = dict(operator=str(func), module_stack=list(self.stack),
                                   inputs=summary((args, kwargs)), outputs=summary(out),
                                   traceback=traceback.format_stack(limit=25))
                raise FloatingPointError(f'首个非有限算子: {func}; 模块: {self.stack}')
        return out


def replay(net, criterion, x, y, output):
    stack, handles = [], []
    for name, module in net.named_modules():
        def enter(module, args, name=name):
            stack.append(name)

        def leave(module, args, result):
            stack.pop()

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave, always_call=True))
    mode = FirstNonfinite(stack)
    net.zero_grad(set_to_none=True)
    try:
        with torch.autograd.set_detect_anomaly(True, check_nan=False), mode:
            prediction = net(x)
            loss = criterion(prediction, y)
            # 反向期间模块 forward hook 不会触发，使用 autograd 节点补充定位。
            pending, visited = [loss.grad_fn], set()
            while pending:
                node = pending.pop()
                if node is None or node in visited:
                    continue
                visited.add(node)

                def before_backward(gradients, node=node):
                    stack.append('backward:' + node.name())

                def after_backward(inputs, outputs):
                    stack.pop()

                handles.append(node.register_prehook(before_backward))
                handles.append(node.register_hook(after_backward))
                pending.extend(next_node for next_node, _ in node.next_functions)
            loss.backward()
    except Exception as error:
        report = mode.report or dict(error=repr(error), traceback=traceback.format_exc())
    else:
        report = dict(error='逐算子重放未复现；检查现场梯度和优化器更新。')
    finally:
        for handle in handles:
            handle.remove()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', default='options/SAR/train_ucan_single_q2.yml')
    parser.add_argument('--checkpoint', default='experiments/ucan_sar_single_q2/models/net_g_120000.pth')
    parser.add_argument('--state', default='experiments/ucan_sar_single_q2/training_states/120000.state')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--output', default='docs/diagnostics/ucan_nonfinite')
    parser.add_argument('--replay-case', help='单 GPU 重放已捕获的现场 .pth，不再执行训练续跑。')
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    opt = yaml_load(args.opt)
    if args.replay_case:
        case = torch.load(args.replay_case, map_location='cpu', weights_only=False)
        net = build_network(opt['network_g']).cuda()
        net.load_state_dict(case['params'])
        criterion = build_loss(opt['train']['pixel_opt']).cuda()
        replay(net, criterion, case['x'].cuda(), case['y'].cuda(),
               Path(args.replay_case).with_suffix('.operator.json'))
        return
    if world != opt['num_gpu']:
        raise ValueError('诊断必须保持原训练 GPU 数量和每卡 batch。')
    set_random_seed(opt['manual_seed'] + rank)
    torch.backends.cudnn.benchmark = True
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    dataopt = dict(opt['datasets']['train'], phase='train')
    dataset = build_dataset(dataopt)
    sampler = EnlargedSampler(dataset, world, rank, dataopt.get('dataset_enlarge_ratio', 1))
    loader = build_dataloader(dataset, dataopt, num_gpu=world, dist=True,
                              sampler=sampler, seed=opt['manual_seed'])
    net = build_network(opt['network_g']).cuda()
    net.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['params'])
    model = DistributedDataParallel(net, device_ids=[local_rank]) if world > 1 else net
    optimopt = opt['train']['optim_g'].copy()
    optimtype = optimopt.pop('type')
    optimizer = getattr(torch.optim, optimtype)(net.parameters(), **optimopt)
    state = torch.load(args.state, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizers'][0])
    scheduleropt = opt['train']['scheduler'].copy()
    schedulertype = scheduleropt.pop('type')
    scheduler = getattr(torch.optim.lr_scheduler, schedulertype)(optimizer, **scheduleropt)
    scheduler.load_state_dict(state['schedulers'][0])
    criterion = build_loss(opt['train']['pixel_opt']).cuda()
    start = iteration = state['iter']
    # 原训练每个 epoch 重建迭代器，worker 的 Python/NumPy seed 固定；
    # 读取并丢弃已完成 batch，恢复 sampler 的位置和每个 worker 的裁剪序列。
    per_epoch = len(loader)
    epoch, skip = divmod(start, per_epoch)
    if epoch != state['epoch']:
        raise ValueError('检查点 epoch 与数据集长度不一致。')
    gradient_flags, gradient_names = [], []
    handles = []
    for name, parameter in net.named_parameters():
        def gradient_hook(gradient, name=name):
            gradient_names.append(name)
            gradient_flags.append(torch.isfinite(gradient).all())
        handles.append(parameter.register_hook(gradient_hook))
    print(f'rank={rank} resume={start} epoch={epoch} skip={skip} steps={args.steps}', flush=True)
    found = False
    while iteration < start + args.steps and not found:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < skip:
                continue
            iteration += 1
            scheduler.step()
            x, y = batch['lq'].cuda(), batch['gt'].cuda()
            optimizer.zero_grad(set_to_none=True)
            gradient_flags.clear()
            gradient_names.clear()
            phase, failure, local_bad_gradients = 'forward', None, []
            # 先让各 rank 完成前向并交换故障标记，避免另一个 rank 卡在反向归约。
            try:
                prediction = model(x)
                loss = criterion(prediction, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('loss 非有限')
            except (ValueError, FloatingPointError) as error:
                failure = repr(error)
            flag = torch.tensor(int(failure is not None), device=x.device)
            if world > 1:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if not flag.item():
                phase = 'backward'
                loss.backward()
                finite = torch.stack(gradient_flags).cpu().tolist()
                local_bad_gradients = [n for n, good in zip(gradient_names, finite) if not good]
                reduced_names = [n for n, p in net.named_parameters() if p.grad is not None]
                reduced_finite = torch.stack([torch.isfinite(p.grad).all() for p in net.parameters()
                                              if p.grad is not None]).cpu().tolist()
                reduced_bad = [n for n, good in zip(reduced_names, reduced_finite) if not good]
                failure = '梯度非有限' if local_bad_gradients or reduced_bad else None
                flag.fill_(int(failure is not None))
                if world > 1:
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if not flag.item():
                phase = 'optimizer'
                optimizer.step()
                finite = torch.stack([torch.isfinite(p).all() for p in net.parameters()])
                failure = '更新后参数非有限' if not bool(finite.all()) else None
                flag.fill_(int(failure is not None))
                if world > 1:
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                prefix = destination / f'iter_{iteration}_rank_{rank}'
                metadata = dict(iteration=iteration, epoch=epoch, batch_index=batch_index,
                                rank=rank, phase=phase, failure=failure,
                                local_bad_gradients=local_bad_gradients, paths=batch['lq_path'])
                prefix.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
                # 现场文件只写诊断目录，正式模型与训练状态保持原样。
                torch.save(dict(params=net.state_dict(), optimizer=optimizer.state_dict(),
                                x=x, y=y, metadata=metadata), prefix.with_suffix('.pth'))
                for handle in handles:
                    handle.remove()
                if failure is not None:
                    replay(net, criterion, x, y, Path(str(prefix) + '_first_operator.json'))
                found = True
                break
            if iteration % 50 == 0:
                rows = [(float(p.sum(-2).abs().min()), name)
                        for name, p in net.named_parameters() if name.endswith(('sin_coef', 'cos_coef'))]
                print(f'rank={rank} iter={iteration} loss={loss.item():.8f} min_fourier_sum={min(rows)}', flush=True)
            if iteration >= start + args.steps:
                break
        epoch += 1
        skip = 0
    if not found and rank == 0:
        print(f'NO_NONFINITE: completed {args.steps} steps from {start}', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
