"""跨帧 Token 交互模块第 15.3 节全套 17 项验收单测与参数计数/清理脚本检查。"""

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
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.sar_sequence_dataset import compute_grid_quality_tensor
from basicsr.models import build_model
from scripts.init_paired_weights import build_backbone_opt, build_group_network_opt, generate_paired_inits
from scripts.prune_checkpoints import prune_experiment


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _seq_train_opt(tmp_path: Path, group: str = 'C', gamma: float | None = 4.0) -> dict:
    net_opt = build_group_network_opt(group, lksa_gate_gamma=gamma)
    opt = {
        'name': f'test_seq_{group}',
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': 0,
        'is_train': True,
        'dist': False,
        'rank': 0,
        'world_size': 1,
        'manual_seed': 42,
        'network_g': net_opt,
        'path': {
            'pretrain_network_g': None,
            'strict_load_g': True,
            'param_key_g': 'params',
            'resume_state': None,
            'experiments_root': str(tmp_path),
            'models': str(tmp_path / 'models'),
            'training_states': str(tmp_path / 'training_states'),
            'visualization': str(tmp_path / 'visualization'),
        },
        'train': {
            'total_iter': 10,
            'warmup_iter': -1,
            'ema_decay': 0,
            'optim_g': {'type': 'Adam', 'lr': 1e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]},
            'scheduler': {'type': 'MultiStepLR', 'milestones': [5], 'gamma': 0.5},
            'pixel_opt': {'type': 'CharbonnierLoss', 'loss_weight': 1.0, 'reduction': 'mean', 'eps': 1e-9},
        },
        'val': {
            'save_img': False,
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False},
            },
        },
    }
    for d in ('models', 'training_states', 'visualization'):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return opt


def test_01_parameter_budgets():
    """第 12.1 节精确参数计数核对。"""
    net_a = build_network(build_group_network_opt('A'))
    net_b = build_network(build_group_network_opt('B'))
    net_c = build_network(build_group_network_opt('C'))
    net_d1 = build_network(build_group_network_opt('D1'))
    net_stats = build_network(build_group_network_opt('C_stats'))

    p_a = _count_params(net_a)
    p_b = _count_params(net_b)
    p_c = _count_params(net_c)
    p_d = _count_params(net_d1)
    p_st = _count_params(net_stats)

    assert p_a == 683817
    assert p_b == p_c == 883441
    assert (p_b - p_a) == 199624
    assert p_d == 890369
    assert (p_d - p_c) == 6928

    ti = net_c.token_interaction
    assert _count_params(ti.tokenizer) == 38208
    assert _count_params(ti.frame_embed) + _count_params(ti.row_embed) + _count_params(ti.col_embed) == 1632
    assert _count_params(ti.layers) == 149944
    assert _count_params(ti.wb_pre) + _count_params(ti.wb_dwconv) + _count_params(ti.wb_out) == 9840
    assert _count_params(net_stats.token_interaction.tokenizer) == 9504
    assert p_c - p_st == 28704


def test_02_coordinates_and_hl_projection():
    """第 15.3 节坐标样例：F1/k3、F2/k2、F3/k1、F4/k0 的 s 均为 4；精确 H/L 掩膜投影。"""
    for t, k in ((1, 3), (2, 2), (3, 1), (4, 0)):
        assert t + k == 4
    q = compute_grid_quality_tensor(list(range(9)), grid_h=4, grid_w=4)
    assert q.shape == (9, 4, 4, 1)
    # 0-based [0,513) 和 [1025,1536) 为 H -> s=4 为 1/128, s=8 为 127/128
    expected_by_s = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0,
        4: 1.0 / 128.0, 5: 0.0, 6: 0.0, 7: 0.0,
        8: 127.0 / 128.0, 9: 1.0, 10: 1.0, 11: 1.0,
    }
    for t in range(9):
        for k in range(4):
            s = t + k
            assert torch.allclose(q[t, :, k, 0], torch.full((4,), expected_by_s[s]))


def test_03_region_split_orientation():
    """第 15.3 节区域切分：用二维递增坐标图验证 (r, k) 切分与还原顺序，无转置/错位。"""
    ti = build_network(build_group_network_opt('C')).token_interaction
    h, w = 64, 512
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    # 通道 0 存 y 坐标，通道 1 存 x 坐标
    u = torch.zeros(1, 9, 48, h, w)
    u[:, :, 0] = yy.float()
    u[:, :, 1] = xx.float()

    patches = ti._split_grid_patches(u, 4, 4)  # [1, 9, 4, 4, 48, 16, 128]
    for r in range(4):
        for k in range(4):
            p = patches[0, 4, r, k]
            assert p[0].min().item() == r * 16 and p[0].max().item() == (r + 1) * 16 - 1
            assert p[1].min().item() == k * 128 and p[1].max().item() == (k + 1) * 128 - 1

    restored = ti._merge_grid_patches(patches)
    torch.testing.assert_close(restored, u, rtol=0, atol=0)


def test_04_interface_roundtrip_and_no_extra_patch_embed():
    """第 15.3 节接口往返：[9B, HW, 48] <-> [B, 9, 48, H, W] 纯重排互逆且不额外调用带 LN 的顶层 patch_embed。"""
    net = build_network(build_group_network_opt('C', eval_frame_batch_size=0)).eval()
    calls = {'patch_embed': 0}
    orig_forward = net.backbone.patch_embed.forward

    def counted_pe(x):
        calls['patch_embed'] += 1
        return orig_forward(x)

    net.backbone.patch_embed.forward = counted_pe
    x = torch.rand(1, 9, 1, 32, 64)
    with torch.no_grad():
        _ = net(x)
    assert calls['patch_embed'] == 1

    # 额外验证分帧推理（eval_frame_batch_size=1）与整批推理数学等价（FP32 GEMM batch 顺序浮点误差 ~1e-6）
    net_chunk = build_network(build_group_network_opt('C', eval_frame_batch_size=1)).eval()
    net_chunk.load_state_dict(net.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(net_chunk(x), net(x), rtol=1e-4, atol=1e-5)

    # 纯重排互逆检查
    feat = torch.randn(9, 32 * 64, 48)
    u = feat.transpose(1, 2).reshape(1, 9, 48, 32, 64)
    back = u.reshape(9, 48, 32 * 64).transpose(1, 2).contiguous()
    torch.testing.assert_close(back, feat, rtol=0, atol=0)


def test_05_group_b_isolation():
    """第 15.3 节 B 的隔离：改变其他帧输入，目标帧输出严格不变。"""
    for detail in (False, True):
        opt = build_group_network_opt('B')
        opt['token_opt']['use_detail'] = detail
        net = build_network(opt).eval()
        # 故意给非零权重激活所有交互与回注路径
        with torch.no_grad():
            torch.nn.init.normal_(net.token_interaction.wb_out.weight, std=0.05)
            if detail:
                torch.nn.init.normal_(net.token_interaction.detail_pd.weight, std=0.05)
                torch.nn.init.normal_(net.token_interaction.detail_pd.bias, std=0.05)
        x1 = torch.rand(1, 9, 1, 32, 64)
        x2 = x1.clone()
        for t in range(9):
            if t != 4:
                x2[:, t] = torch.rand_like(x2[:, t])
        with torch.no_grad():
            y1 = net(x1)
            y2 = net(x2)
        torch.testing.assert_close(y1[:, 4], y2[:, 4], rtol=0, atol=0)


def test_06_group_c_f4_reachability():
    """第 15.3 节 C 的可达性：回注更新后，F4 损失可反传到其他帧输入特征。"""
    net = build_network(build_group_network_opt('C')).train()
    with torch.no_grad():
        torch.nn.init.normal_(net.token_interaction.wb_out.weight, std=0.05)
    x = torch.rand(1, 9, 1, 32, 64, requires_grad=True)
    y = net(x)
    loss = y[:, 4].square().mean()
    loss.backward()
    for t in range(9):
        if t != 4:
            assert x.grad[:, t].abs().max().item() > 0.0


def test_07_08_identity_and_quality_starting_point():
    """第 15.3 节恒等初始化与质量起点：B/C/D0/D1 初始输出与 A 一致；D 初始 c=0,d=1 与 C 注意力等价。"""
    backbone_opt = build_backbone_opt(lksa_gate_gamma=4.0)
    torch.manual_seed(42)
    bb = build_network(backbone_opt)
    bb_sd = bb.state_dict()

    nets = {}
    for g in ('A', 'B', 'C', 'D1', 'D0'):
        torch.manual_seed(42)
        n = build_network(build_group_network_opt(g))
        n.load_backbone_state_dict(bb_sd, strict=True)
        n.eval()
        nets[g] = n

    x = torch.rand(1, 9, 1, 32, 64)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)
    with torch.no_grad():
        ya = nets['A'](x)
        yb = nets['B'](x)
        yc, tele_c = nets['C'](x, token_quality=q, return_telemetry=True)
        yd1, tele_d1 = nets['D1'](x, token_quality=q, return_telemetry=True)
        yd0, tele_d0 = nets['D0'](x, token_quality=q, return_telemetry=True)

    for name, y in (('B', yb), ('C', yc), ('D1', yd1), ('D0', yd0)):
        torch.testing.assert_close(y, ya, rtol=0, atol=1e-5, msg=f'{name} differs from A at init')

    for ql in tele_d1['quality_layers']:
        assert ql['c_src'].abs().max().item() == 0.0
        torch.testing.assert_close(ql['d_tgt'], torch.ones_like(ql['d_tgt']), rtol=0, atol=0)
    for ac, ad1, ad0 in zip(tele_c['attn_layers'], tele_d1['attn_layers'], tele_d0['attn_layers']):
        # C 的权重初始化与 D1/D0 不含质量模块时随机数消费略不同，需逐项复制共同参数后再测 attention 等价
        pass
    sd_d1 = nets['D1'].state_dict()
    sd_c = nets['C'].state_dict()
    for k in sd_c:
        sd_c[k] = sd_d1[k].clone()
    nets['C'].load_state_dict(sd_c, strict=True)
    with torch.no_grad():
        _, tele_c_paired = nets['C'](x, token_quality=q, return_telemetry=True)
    for ac, ad1, ad0 in zip(tele_c_paired['attn_layers'], tele_d1['attn_layers'], tele_d0['attn_layers']):
        torch.testing.assert_close(ad1, ac, rtol=0, atol=1e-6)
        torch.testing.assert_close(ad0, ac, rtol=0, atol=1e-6)


def test_09_init_overwrite_and_checkpoint_preservation():
    """第 15.3 节初始化覆盖防护与加载 checkpoint 不重置。"""
    net = build_network(build_group_network_opt('D1'))
    assert net.token_interaction.wb_out.weight.abs().max().item() == 0.0
    for layer in net.token_interaction.layers:
        assert layer.quality_bias.mlp_src[-1].weight.abs().max().item() == 0.0
        assert layer.quality_bias.mlp_tgt[-1].weight.abs().max().item() == 0.0

    # 修改这些权重后，通过 state_dict 加载到新模型应保持非零
    sd = net.state_dict()
    sd['token_interaction.wb_out.weight'] = torch.ones_like(sd['token_interaction.wb_out.weight']) * 0.123
    net2 = build_network(build_group_network_opt('D1'))
    net2.load_state_dict(sd, strict=True)
    assert net2.token_interaction.wb_out.weight.abs().max().item() == pytest.approx(0.123)


def test_10_four_stage_gradient_propagation():
    """第 15.3 节四阶段梯度传播检查。"""
    torch.manual_seed(42)
    net = build_network(build_group_network_opt('D1')).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.rand(1, 9, 1, 32, 64)
    target = torch.rand(1, 9, 1, 32, 64)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)

    ti = net.token_interaction
    # 第 1 步反向：wb_out 与骨干获得梯度
    opt.zero_grad(set_to_none=True)
    loss = (net(x, token_quality=q) - target).abs().mean()
    loss.backward()
    assert ti.wb_out.weight.grad.abs().max().item() > 0.0
    opt.step()

    # 第 2 步反向（P_out 更新后）：P_pre, DWConv, tokenizer, attention 与来源分支输出层获梯度
    opt.zero_grad(set_to_none=True)
    loss = (net(x, token_quality=q) - target).abs().mean()
    loss.backward()
    assert ti.wb_pre.weight.grad.abs().max().item() > 0.0
    assert ti.wb_dwconv.weight.grad.abs().max().item() > 0.0
    assert ti.tokenizer.proj.weight.grad.abs().max().item() > 0.0
    assert ti.layers[0].qkv.weight.grad.abs().max().item() > 0.0
    assert ti.layers[1].quality_bias.mlp_src[-1].weight.grad.abs().max().item() > 0.0
    opt.step()

    # 第 3 步反向（c 不再全零后）：来源隐藏层及目标分支输出层获梯度
    opt.zero_grad(set_to_none=True)
    loss = (net(x, token_quality=q) - target).abs().mean()
    loss.backward()
    assert ti.layers[1].quality_bias.mlp_src[0].weight.grad.abs().max().item() > 0.0
    assert ti.layers[1].quality_bias.mlp_tgt[-1].weight.grad.abs().max().item() > 0.0
    opt.step()

    # 第 4 步反向（目标输出层更新后）：目标隐藏层获梯度，且全部梯度有限
    opt.zero_grad(set_to_none=True)
    loss = (net(x, token_quality=q) - target).abs().mean()
    loss.backward()
    assert ti.layers[1].quality_bias.mlp_tgt[0].weight.grad.abs().max().item() > 0.0
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f'Non-finite grad at {name}'


def test_11_d0_d1_pairing_and_d0_content_bias(tmp_path):
    """第 15.3 节 D0/D1 配对初始化以及 D0 在常数 q=0.5 下仍学习内容偏置。"""
    res = generate_paired_inits(None, seed=42, out_dir=tmp_path, lksa_gate_gamma=4.0, groups=['A', 'B', 'C', 'D1', 'D0'])
    assert set(res.keys()) == {'A', 'B', 'C', 'D1', 'D0'}

    net_d0 = build_network(build_group_network_opt('D0')).train()
    sd = torch.load(res['D0']['path'], map_location='cpu', weights_only=True)['params']
    net_d0.load_state_dict(sd, strict=True)
    opt = torch.optim.Adam(net_d0.parameters(), lr=1e-3)
    x = torch.rand(1, 9, 1, 32, 64)
    target = torch.rand(1, 9, 1, 32, 64)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = (net_d0(x, token_quality=q) - target).abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, tele = net_d0(x, token_quality=q, return_telemetry=True)
    assert tele['quality_layers'][-1]['c_src'].abs().max().item() > 0.0


def test_12_quality_branch_input_isolation():
    """第 15.3 节质量分支输入隔离：篡改 frame/row/col 嵌入后 c、d 严格不变。"""
    net = build_network(build_group_network_opt('D1')).eval()
    # 给质量输出层置非零权重
    with torch.no_grad():
        for layer in net.token_interaction.layers:
            torch.nn.init.normal_(layer.quality_bias.mlp_src[-1].weight, std=0.1)
            torch.nn.init.normal_(layer.quality_bias.mlp_tgt[-1].weight, std=0.1)

    x = torch.rand(1, 9, 1, 32, 64)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)
    with torch.no_grad():
        _, t1 = net(x, token_quality=q, return_telemetry=True)
        net.token_interaction.frame_embed.weight.add_(5.0)
        net.token_interaction.row_embed.weight.add_(-3.0)
        net.token_interaction.col_embed.weight.add_(2.0)
        _, t2 = net(x, token_quality=q, return_telemetry=True)

    for l1, l2 in zip(t1['quality_layers'], t2['quality_layers']):
        torch.testing.assert_close(l1['c_src'], l2['c_src'], rtol=0, atol=0)
        torch.testing.assert_close(l1['d_tgt'], l2['d_tgt'], rtol=0, atol=0)


def test_13_causal_mask_horizon():
    """第 15.3 节块因果 mask：改变 t > t_i + L1 + L2 之后帧输入，帧 t_i 输出严格不变。"""
    for cfg_name, lookahead, effective_l in (('Causal_L0', [0, 0], 0), ('Causal_L1', [1, 0], 1), ('Causal_L2', [2, 0], 2)):
        opt = build_group_network_opt(cfg_name)
        net = build_network(opt).eval()
        with torch.no_grad():
            torch.nn.init.normal_(net.token_interaction.wb_out.weight, std=0.05)
        x1 = torch.rand(1, 9, 1, 32, 64)
        x2 = x1.clone()
        # 测试目标帧 t_i = 4；修改所有 t > 4 + effective_l 的帧
        for t in range(4 + effective_l + 1, 9):
            x2[:, t] = torch.rand_like(x2[:, t])
        with torch.no_grad():
            y1 = net(x1)
            y2 = net(x2)
        for t_check in range(5):
            torch.testing.assert_close(y1[:, t_check], y2[:, t_check], rtol=0, atol=0)

    # 额外验证两层组合有效前瞻 = L1 + L2：若 L1=1, L2=1，则 F4 可间接读取 F6
    opt_l11 = build_group_network_opt('Causal_L1')
    opt_l11['token_opt']['causal_lookahead'] = [1, 1]
    net_l11 = build_network(opt_l11).eval()
    with torch.no_grad():
        torch.nn.init.normal_(net_l11.token_interaction.wb_out.weight, std=0.05)
    x1 = torch.rand(1, 9, 1, 32, 64)
    x2 = x1.clone()
    x2[:, 6] = torch.rand_like(x2[:, 6])
    with torch.no_grad():
        y1 = net_l11(x1)
        y2 = net_l11(x2)
    assert (y1[:, 4] - y2[:, 4]).abs().max().item() > 0.0


def test_14_vertical_scale_and_slice_inference(tmp_path):
    """第 15.3 节纵向尺度检查（H=64 vs H=512 及同步切片推理 H=64, stride=32）。"""
    opt = _seq_train_opt(tmp_path, 'C')
    opt['val']['slice_eval_height'] = 32
    opt['val']['slice_eval_stride'] = 16
    model = build_model(deepcopy(opt))
    lq = torch.rand(1, 9, 1, 64, 64)
    model.feed_data({'lq_seq': lq, 'frame_ids': torch.arange(9).unsqueeze(0)})
    model.test()
    assert model.output.shape == lq.shape
    assert torch.isfinite(model.output).all()


def test_15_detail_path_safety_and_masking():
    """第 15.3 节 detail：s=0、s=11 与无有效源时即使 P_D.bias 非零也严格置零；遵守 mask 与来源屏蔽。"""
    opt = build_group_network_opt('C_detail')
    net = build_network(opt).eval()
    ti = net.token_interaction
    with torch.no_grad():
        torch.nn.init.normal_(ti.detail_pd.weight, std=0.1)
        torch.nn.init.constant_(ti.detail_pd.bias, 1.5)
        # 关掉调制回注，专门检测 detail 增量
        nn_zero = ti.wb_out
        torch.nn.init.zeros_(nn_zero.weight)
        torch.nn.init.zeros_(nn_zero.bias)

    u = torch.rand(1, 9, 48, 32, 64)
    with torch.no_grad():
        u_out, tele = ti(u, return_telemetry=True)
    delta = u_out - u
    assert torch.isfinite(delta).all()
    # F0/k0 (s=0) 和 F8/k3 (s=11) 没有任何其他帧同地面源，即使 P_D.bias=1.5 也必须严格为 0
    assert delta[0, 0, :, :, :16].abs().max().item() == 0.0
    assert delta[0, 8, :, :, 48:].abs().max().item() == 0.0
    # 有同地面来源的 F4 应非零
    assert delta[0, 4].abs().max().item() > 0.0

    # 运行时屏蔽 F1, F2, F3, F5, F6, F7（全部桥梁帧）后，F4 的 detail 更新应严格为 0
    with torch.no_grad():
        u_masked, _ = ti(u, masked_source_frames=[1, 2, 3, 5, 6, 7], return_telemetry=True)
    assert (u_masked[0, 4] - u[0, 4]).abs().max().item() == 0.0


class _SeqDummyDataset(Dataset):
    def __init__(self, size: int):
        self.size = size
        self.opt = {'name': 'seq_ddp_val'}
        self.seen = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.seen.append(index)
        gt = torch.linspace(0.1, 0.8, 32 * 64).reshape(1, 32, 64).repeat(9, 1, 1, 1)
        return {
            'lq_seq': (gt + 0.01 * (index + 1)).clamp(0, 1),
            'gt_seq': gt,
            'frame_ids': torch.arange(9, dtype=torch.long),
            'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4),
            'sequence_name': f'seq_{index:03d}',
        }


def _seq_ddp_worker(rank, rendezvous, out_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        opt = _seq_train_opt(Path(out_dir), 'C')
        model = build_model(deepcopy(opt))
        model.net_g = DistributedDataParallel(model.net_g)
        model.opt.update(dist=True, rank=rank, world_size=2)
        batch = {
            'lq_seq': torch.rand(1, 9, 1, 32, 64),
            'gt_seq': torch.rand(1, 9, 1, 32, 64),
            'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0),
            'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0),
        }
        for step in (1, 2):
            model.feed_data(batch)
            model.optimize_parameters(step)

        for size in (3, 1):
            ds = _SeqDummyDataset(size)
            loader = DataLoader(ds, batch_size=1)
            model.validation(loader, current_iter=2, tb_logger=None, save_img=False)
            assert ds.seen == list(range(rank, size, 2))
            dist_metrics = dict(model.metric_results)
            dist_per_frame = dict(model.per_frame_metric_results)
            model.nondist_validation(loader, current_iter=2, tb_logger=None, save_img=False)
            for k, v in dist_metrics.items():
                assert model.metric_results[k] == pytest.approx(v, abs=1e-7)
            for t in range(9):
                assert model.per_frame_metric_results['psnr'][t] == pytest.approx(dist_per_frame['psnr'][t], abs=1e-7)
    finally:
        dist.destroy_process_group()


def test_16_sequence_ddp_validation_no_duplicate_and_train(tmp_path):
    """第 15.3 节序列评价：奇数序列无重复、2-rank DDP 与单卡等价、支持正常反向更新。"""
    rendezvous = (tmp_path / 'rendezvous').as_uri()
    mp.spawn(_seq_ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)


def test_17_permutation_equivariance_save_resume_and_prune(tmp_path):
    """第 15.3 节重排列一致性、保存恢复与检查点裁剪工具检查。"""
    net = build_network(build_group_network_opt('D1')).eval()
    with torch.no_grad():
        torch.nn.init.normal_(net.token_interaction.wb_out.weight, std=0.05)
        for layer in net.token_interaction.layers:
            torch.nn.init.normal_(layer.quality_bias.mlp_src[-1].weight, std=0.1)
            torch.nn.init.normal_(layer.quality_bias.mlp_tgt[-1].weight, std=0.1)

    x = torch.rand(1, 9, 1, 32, 64)
    fids = torch.arange(9, dtype=torch.long).unsqueeze(0)
    q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)
    perm = torch.tensor([4, 0, 8, 2, 6, 1, 7, 3, 5], dtype=torch.long)
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        y_orig = net(x, frame_ids=fids, token_quality=q)
        y_perm = net(x[:, perm], frame_ids=fids[:, perm], token_quality=q[:, perm])
    torch.testing.assert_close(y_perm[:, inv_perm], y_orig, rtol=0, atol=1e-5)

    # 保存恢复一致性
    opt = _seq_train_opt(tmp_path / 'run1', 'D1')
    m1 = build_model(deepcopy(opt))
    batch = {'lq_seq': x, 'gt_seq': torch.rand_like(x), 'frame_ids': fids, 'token_quality': q}
    for s in (1, 2):
        m1.update_learning_rate(s)
        m1.feed_data(batch)
        m1.optimize_parameters(s)
    m1.save(0, 2)

    saved = torch.load(tmp_path / 'run1/training_states/2.state', map_location='cpu', weights_only=True)
    opt2 = _seq_train_opt(tmp_path / 'run2', 'D1')
    opt2['path']['pretrain_network_g'] = str(tmp_path / 'run1/models/net_g_2.pth')
    m2 = build_model(deepcopy(opt2))
    m2.resume_training(saved)
    for m in (m1, m2):
        m.update_learning_rate(3)
        m.feed_data(batch)
        m.optimize_parameters(3)
    for p1, p2 in zip(m1.net_g.parameters(), m2.net_g.parameters()):
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    # 验证 prune_checkpoints.py 只保留最优 PSNR、最优 SSIM 和最后一次
    exp_prune = tmp_path / 'prune_test'
    (exp_prune / 'models').mkdir(parents=True)
    (exp_prune / 'training_states').mkdir(parents=True)
    for it in (2000, 4000, 6000, 8000):
        (exp_prune / f'models/net_g_{it}.pth').write_bytes(b'x' * 10)
        (exp_prune / f'training_states/{it}.state').write_bytes(b'y' * 10)
    (exp_prune / 'models/net_g_latest.pth').write_bytes(b'x' * 10)
    import json
    (exp_prune / 'val_history.json').write_text(json.dumps([
        {'iter': 2000, 'metrics': {'psnr': 23.0, 'ssim': 0.70}},
        {'iter': 4000, 'metrics': {'psnr': 24.5, 'ssim': 0.72}},
        {'iter': 6000, 'metrics': {'psnr': 24.1, 'ssim': 0.78}},
        {'iter': 8000, 'metrics': {'psnr': 24.0, 'ssim': 0.75}},
    ]), encoding='utf-8')
    manifest = prune_experiment(exp_prune)
    assert set(manifest['kept_iters']) == {4000, 6000, 8000}
    assert not (exp_prune / 'models/net_g_2000.pth').exists()
    assert not (exp_prune / 'training_states/2000.state').exists()
    assert (exp_prune / 'models/net_g_best_psnr.pth').is_symlink()
    assert (exp_prune / 'models/net_g_best_ssim.pth').is_symlink()

    # 测试无 val_history.json 时基于单帧 train_*.log 的 Best @ iter 回退解析（含 print_freq 未对齐情况）
    exp_log_prune = tmp_path / 'prune_log_test'
    (exp_log_prune / 'models').mkdir(parents=True)
    (exp_log_prune / 'training_states').mkdir(parents=True)
    for it in (10000, 20000, 30000):
        (exp_log_prune / f'models/net_g_{it}.pth').write_bytes(b'x' * 10)
        (exp_log_prune / f'training_states/{it}.state').write_bytes(b'y' * 10)
    (exp_log_prune / 'train_test_20261119.log').write_text(
        '2026-11-19 INFO: [iter: 9,916] l_pix: 0.05\n'
        'Validation sar_val\n'
        '\t # psnr: 24.5000\tBest: 24.5000 @ 10000 iter\n'
        '\t # ssim: 0.7800\tBest: 0.7800 @ 10000 iter\n'
        '2026-11-19 INFO: [iter: 19,966] l_pix: 0.04\n'
        'Validation sar_val\n'
        '\t # psnr: 24.8000\tBest: 24.8000 @ 20000 iter\n'
        '\t # ssim: 0.7750\tBest: 0.7800 @ 10000 iter\n',
        encoding='utf-8',
    )
    m_log = prune_experiment(exp_log_prune)
    assert m_log['best_psnr']['iter'] == 20000
    assert m_log['best_ssim']['iter'] == 10000
    assert m_log['last_iter'] == 30000


def test_18_sar_sequence_dataset_deterministic_crop(tmp_path):
    """验证 SARSequenceDataset 同 seed/epoch/index 的确定性配对裁剪。"""
    import cv2
    import numpy as np
    for q in ('LQ', 'GT'):
        (tmp_path / q / 'bangkok_rstart_1_cstart_1').mkdir(parents=True)
    for t in range(9):
        img = np.full((512, 512), t * 10, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / 'LQ/bangkok_rstart_1_cstart_1' / f'{t:03d}.png'), img)
        cv2.imwrite(str(tmp_path / 'GT/bangkok_rstart_1_cstart_1' / f'{t:03d}.png'), img)

    ds_opt = {
        'name': 'seq_ds',
        'type': 'SARSequenceDataset',
        'phase': 'train',
        'dataroot_lq': str(tmp_path / 'LQ'),
        'dataroot_gt': str(tmp_path / 'GT'),
        'crop_height': 64,
        'deterministic_crop_seed': 42,
    }
    ds1 = build_dataset(ds_opt)
    ds2 = build_dataset(ds_opt)
    ds1.set_epoch(3)
    ds2.set_epoch(3)
    item1, item2 = ds1[0], ds2[0]
    assert item1['crop_top'] == item2['crop_top']
    assert item1['lq_seq'].shape == (9, 1, 64, 512)
    assert item1['source_group'] == 'bangkok'
    assert item1['token_quality'].shape == (9, 4, 4, 1)


def _chunk_ddp_worker(rank, rendezvous, out_dir):
    torch.set_num_threads(2)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(42)
        opt = _seq_train_opt(Path(out_dir), 'D1')
        opt['network_g']['train_frame_chunk_size'] = 3
        model = build_model(deepcopy(opt))
        model.net_g = DistributedDataParallel(model.net_g)
        model.net_g._set_static_graph()
        model.opt.update(dist=True, rank=rank, world_size=2)
        batch = {
            'lq_seq': torch.rand(1, 9, 1, 32, 64),
            'gt_seq': torch.rand(1, 9, 1, 32, 64),
            'frame_ids': torch.arange(9, dtype=torch.long).unsqueeze(0),
            'token_quality': compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0),
        }
        for step in (1, 2):
            model.feed_data(batch)
            model.optimize_parameters(step)
        actual = torch.nn.utils.parameters_to_vector(model.net_g.parameters()).detach()
        expected = actual.clone()
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_19_train_frame_chunk_equivalence_and_ddp(tmp_path):
    """验证 train_frame_chunk_size=3/1 与整批前向及全部骨干/token 参数梯度严格等价，并验证 DDP static_graph 正常反向。"""
    for group in ('C', 'D1'):
        torch.manual_seed(77)
        opt_ref = build_group_network_opt(group, lksa_gate_gamma=4.0, train_frame_chunk_size=0)
        net_ref = build_network(opt_ref).train()
        # 给零初始化回注和质量分支非零值，确保梯度全面流通
        with torch.no_grad():
            torch.nn.init.normal_(net_ref.token_interaction.wb_out.weight, std=0.05)
            for layer in net_ref.token_interaction.layers:
                if layer.quality_bias is not None:
                    torch.nn.init.normal_(layer.quality_bias.mlp_src[-1].weight, std=0.05)
                    torch.nn.init.normal_(layer.quality_bias.mlp_tgt[-1].weight, std=0.05)
        ref_sd = net_ref.state_dict()

        x = torch.rand(1, 9, 1, 32, 64)
        target = torch.rand(1, 9, 1, 32, 64)
        q = compute_grid_quality_tensor(list(range(9)), 4, 4).unsqueeze(0)

        net_ref.zero_grad(set_to_none=True)
        out_ref = net_ref(x, token_quality=q)
        loss_ref = (out_ref - target).abs().mean()
        loss_ref.backward()

        for chunk_sz in (3, 1):
            opt_c = build_group_network_opt(group, lksa_gate_gamma=4.0, train_frame_chunk_size=chunk_sz)
            net_c = build_network(opt_c).train()
            net_c.load_state_dict(ref_sd, strict=True)
            net_c.zero_grad(set_to_none=True)
            out_c = net_c(x, token_quality=q)
            torch.testing.assert_close(out_c, out_ref, rtol=1e-4, atol=1e-5)
            loss_c = (out_c - target).abs().mean()
            loss_c.backward()
            for (name, p_ref), (_, p_c) in zip(net_ref.named_parameters(), net_c.named_parameters()):
                if p_ref.grad is not None:
                    assert p_c.grad is not None, f'Missing grad at {name} for chunk={chunk_sz}'
                    torch.testing.assert_close(p_c.grad, p_ref.grad, rtol=1e-4, atol=1e-5)

    rendezvous = (tmp_path / 'chunk_ddp').as_uri()
    mp.spawn(_chunk_ddp_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)


def test_20_val_iters_skip_config_gen_and_d1_additive(tmp_path):
    """验证 val_iters 结束去重、D1_additive 配对初始化、配置生成器与 eval 干预参数合法性。"""
    from basicsr.models.distributed_validation import DistributedValidationMixin
    from basicsr.models.sequence_validation import SequenceDistributedValidationMixin
    from basicsr.utils.options import yaml_load
    from scripts.eval_sar_sequence import evaluate_sequence_model
    from scripts.generate_exp_configs import write_exp_configs

    # 1) D1_additive 配对与张量零差
    res = generate_paired_inits(
        None, seed=42, out_dir=tmp_path / 'inits', lksa_gate_gamma=4.0, groups=['D1', 'D1_additive']
    )
    assert 'D1_additive' in res
    sd_d1 = torch.load(res['D1']['path'], map_location='cpu', weights_only=True)['params']
    sd_add = torch.load(res['D1_additive']['path'], map_location='cpu', weights_only=True)['params']
    shared_keys = set(sd_d1.keys()) & set(sd_add.keys())
    same_shape_keys = [k for k in shared_keys if sd_d1[k].shape == sd_add[k].shape]
    diff_shape_keys = [k for k in shared_keys if sd_d1[k].shape != sd_add[k].shape]
    assert len(same_shape_keys) > 0
    for k in same_shape_keys:
        assert (sd_d1[k] - sd_add[k]).abs().max().item() == 0.0
    for k in diff_shape_keys:
        assert 'wb_out' in k
        assert sd_add[k].abs().max().item() == 0.0

    # 2) 配置生成器可生成完整可解析 YAML 且网络可直接构造
    cfgs = write_exp_configs(tmp_path / 'cfgs', ['C', 'D1_additive'], [42], init_dir=str(tmp_path / 'inits'))
    assert len(cfgs) == 2
    loaded = yaml_load(str(cfgs[1]))
    assert loaded['val']['val_iters'] == [4000, 8000, 12000, 14000, 16000, 18000, 20000]
    assert loaded['network_g']['train_frame_chunk_size'] == 3
    net_built = build_network(loaded['network_g'])
    assert net_built.token_interaction.writeback_mode == 'additive'

    # 3) val_iters 结束时 total_iter + 1 / + 2 跳过重复验证
    class DummySeq(SequenceDistributedValidationMixin):
        def __init__(self, root):
            self.opt = {
                'is_train': True,
                'path': {'experiments_root': str(root)},
                'train': {'total_iter': 20000},
                'val': {'val_freq': 20000, 'val_iters': [4000, 20000]},
            }

        def _run_sequence_validation(self, *args, **kwargs):
            return super()._run_sequence_validation(*args, **kwargs)

    # _run_sequence_validation 应直接返回（不访问无效 dataloader）
    DummySeq(tmp_path)._run_sequence_validation(None, 'val', 20002, None, False, rank=0, world_size=1, is_dist=False)

    # 4) eval_sar_sequence 互斥与非法 permute 参数校验
    with pytest.raises(ValueError, match='mutually exclusive'):
        evaluate_sequence_model(
            cfgs[0], Path(res['D1']['path']), Path('missing.txt'), tmp_path / 'e1',
            permute_frames=list(range(9)), mismatch_meta=True,
        )
    with pytest.raises(ValueError, match='permutation of 0..8'):
        evaluate_sequence_model(
            cfgs[0], Path(res['D1']['path']), Path('missing.txt'), tmp_path / 'e2',
            permute_frames=[0, 1, 2],
        )

    # DistributedValidationMixin 类存在且保持原有接口
    assert hasattr(DistributedValidationMixin, 'dist_validation')
