"""UCAN 的恢复分支、原生保存恢复和不等长 DDP 验证检查，全部在 CPU 执行。"""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from basicsr.archs import build_network
from basicsr.models import build_model
from basicsr.models.ucan_model import UCANModel
from basicsr.utils.options import yaml_load


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def network_options():
    return yaml_load('options/SAR/train_ucan_single_q2.yml')['network_g']


def training_options(output):
    opt = yaml_load('options/SAR/train_ucan_single_q2.yml')
    opt.update(is_train=True, dist=False, num_gpu=0, rank=0, world_size=1)
    for name in ('models', 'training_states', 'visualization'):
        path = Path(output) / name
        path.mkdir(parents=True, exist_ok=True)
        opt['path'][name] = str(path)
    return opt


@pytest.mark.parametrize('suffix', ['', '_q2'])
def test_configuration_matches_sar_protocol(suffix):
    train = yaml_load(f'options/SAR/train_ucan_single{suffix}.yml')
    test = yaml_load(f'options/SAR/test_ucan_single{suffix}.yml')
    reference = yaml_load(f'options/SAR/train_swinir_single{suffix}.yml')
    assert train['model_type'] == test['model_type'] == 'UCANModel'
    assert train['network_g'] == test['network_g']
    assert train['datasets'] == reference['datasets']
    assert train['train'] == reference['train']
    assert train['val'] == reference['val']
    assert train['scale'] == test['scale'] == 1
    assert Path(train['final_test_opt']).name == f'test_ucan_single{suffix}.yml'
    assert test['path']['pretrain_network_g'] == f'experiments/{train["name"]}/models/net_g_latest.pth'
    for options in (train, test):
        for dataset in options['datasets'].values():
            assert dataset['type'] == 'SARSingleDataset'
            assert dataset['dataroot_lq'].endswith('/LQ')
            assert dataset['dataroot_gt'].endswith('/GT')
            assert Path(dataset['split_file']).is_file()


def test_all_parameters_receive_finite_gradients():
    net = build_network(network_options())
    image = torch.rand(1, 1, 32, 32)
    output = net(image)
    assert output.shape == image.shape
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    for name, param in net.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    # 位置编码随模型设备迁移，但不会向官方检查点增加持久键。
    buffers = {key: value for key, value in net.named_buffers() if key.endswith('inv_freq')}
    assert len(buffers) == 16
    assert all(value.device.type == 'cpu' for value in buffers.values())
    assert not any(key.endswith('inv_freq') for key in net.state_dict())


@pytest.mark.parametrize('shape', [(128, 128), (512, 512), (37, 53)])
def test_full_frame_and_non_window_size(shape):
    net = build_network(network_options()).eval()
    image = torch.rand(1, 1, *shape)
    with torch.no_grad():
        output = net(image)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_checkpointing_preserves_gradients():
    options = network_options()
    direct = build_network(options)
    checkpointed = build_network(dict(options, use_checkpoint=True))
    checkpointed.load_state_dict(direct.state_dict(), strict=True)
    image = torch.rand(1, 1, 32, 32)
    expected, actual = direct(image), checkpointed(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for first, second in zip(direct.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(second.grad, first.grad)


class RecordingNet(torch.nn.Module):

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.seen = False

    def forward(self, image):
        assert not self.training and not torch.is_grad_enabled()
        self.seen = True
        if self.fail:
            raise RuntimeError('intentional failure')
        return image + 1


@pytest.mark.parametrize('training,ema,fail', [(True, False, False), (False, True, False), (True, False, True)])
def test_inference_mode_and_ema(training, ema, fail):
    model = UCANModel.__new__(UCANModel)
    model.net_g = RecordingNet(fail).train(training)
    if ema:
        model.net_g_ema = RecordingNet(fail).train(training)
    model.lq = torch.rand(1, 1, 37, 53)
    if fail:
        with pytest.raises(RuntimeError, match='intentional failure'):
            model.test()
    else:
        model.test()
        torch.testing.assert_close(model.output, model.lq + 1)
    selected = model.net_g_ema if ema else model.net_g
    assert selected.training == training
    assert selected.seen
    assert model.net_g.seen == (not ema)


def test_native_save_resume_continues_same_update(tmp_path):
    opt = training_options(tmp_path)
    opt['train']['scheduler']['milestones'] = [2]
    model = build_model(deepcopy(opt))
    batch = dict(lq=torch.rand(1, 1, 32, 32), gt=torch.rand(1, 1, 32, 32))
    for step in (1, 2):
        model.update_learning_rate(step)
        model.feed_data(batch)
        model.optimize_parameters(step)
    model.save(0, 2)
    saved = torch.load(tmp_path / 'training_states/2.state', map_location='cpu', weights_only=True)
    opt['path']['pretrain_network_g'] = str(tmp_path / 'models/net_g_2.pth')
    restored = build_model(deepcopy(opt))
    restored.resume_training(saved)
    for current in (model, restored):
        current.update_learning_rate(3)
        current.feed_data(batch)
        current.optimize_parameters(3)
    for first, second in zip(model.net_g.parameters(), restored.net_g.parameters()):
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert restored.schedulers[0].state_dict() == model.schedulers[0].state_dict()
    assert restored.optimizer_g.param_groups[0]['lr'] == opt['train']['optim_g']['lr'] * 0.5


class ValidationFrames(Dataset):

    def __init__(self, size):
        self.size = size
        self.opt = dict(name='ucan_cpu_ddp')
        self.seen = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.seen.append(index)
        gt = torch.linspace(0.1, 0.8, 32 * 32).reshape(1, 32, 32)
        return dict(lq=gt + 0.01 * (index + 1), gt=gt, lq_path=f'frame_{index}.png')


def ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42 + rank)
        opt = training_options(output)
        model = build_model(deepcopy(opt))
        # BasicSR 的生产 DDP 包装器依赖 CUDA，此处仅替换包装步骤来验证 CPU 通信。
        model.net_g = DistributedDataParallel(model.net_g)
        model.opt.update(dist=True, rank=rank, world_size=2)
        batch = dict(lq=torch.rand(1, 1, 32, 32), gt=torch.rand(1, 1, 32, 32))
        for step in (1, 2):
            model.feed_data(batch)
            model.optimize_parameters(step)
        for size in (3, 1):
            dataset = ValidationFrames(size)
            loader = DataLoader(dataset, batch_size=1)
            model.validation(loader, current_iter=2, tb_logger=None, save_img=False)
            assert dataset.seen == list(range(rank, size, 2))
            assert model.net_g.training and model.net_g.module.training
            metrics = dict(model.metric_results)
            model.nondist_validation(loader, current_iter=2, tb_logger=None, save_img=False)
            for name, value in metrics.items():
                assert model.metric_results[name] == pytest.approx(value, abs=1e-8)
        model.feed_data(batch)
        model.optimize_parameters(3)
        actual = torch.nn.utils.parameters_to_vector(model.net_g.parameters()).detach()
        expected = actual.clone()
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_uneven_and_empty_validation_then_train(tmp_path):
    rendezvous = (tmp_path / 'rendezvous').as_uri()
    mp.spawn(ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
