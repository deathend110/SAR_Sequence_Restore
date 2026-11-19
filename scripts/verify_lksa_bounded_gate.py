"""验证 LKSA 有界门控（gamma=None/2/4）在失稳现场与正常验证样本上的数值行为（指导书第 10.2 节）。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from basicsr.archs import build_network
from basicsr.archs.ucan_arch import LKSA
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.options import yaml_load


def inspect_failure_case(case_path: Path, net_opt: dict, loss_opt: dict, gamma: float | None, device: torch.device):
    """在已捕获的非有限现场权重和 batch 上测试指定 gamma 的前向、反向及 5 步微更新稳定性。"""
    case = torch.load(case_path, map_location='cpu', weights_only=False)
    opt = dict(net_opt, lksa_gate_gamma=gamma)
    net = build_network(opt).to(device)
    net.load_state_dict(case['params'], strict=True)
    net.train()
    criterion = build_loss(loss_opt).to(device)
    x = case['x'].to(device)
    y = case['y'].to(device)

    gate_max_vals = []
    lksa_out_max_vals = []
    handles = []

    def make_hook(module: LKSA):
        def _hook(_mod, inputs, output):
            inp = inputs[0].detach()
            with torch.no_grad():
                yy = module.conv_big_v(
                    module.conv_big_h(module.conv_dv(module.conv_dh(module.conv_v(module.conv_h(inp)))))
                )
                raw_gate = yy + module.conv3(inp)
                if module.gate_gamma is not None:
                    bounded = module.gate_gamma * torch.tanh(raw_gate / module.gate_gamma)
                    gate_max_vals.append(float(bounded.abs().max()))
                else:
                    gate_max_vals.append(float(raw_gate.abs().max()))
            lksa_out_max_vals.append(float(output.detach().abs().max()) if torch.isfinite(output).all() else float('inf'))

        return _hook

    for mod in net.modules():
        if isinstance(mod, LKSA):
            handles.append(mod.register_forward_hook(make_hook(mod)))

    result = {
        'case': case_path.name,
        'gamma': gamma,
        'forward_ok': False,
        'forward_error': None,
        'loss': None,
        'grad_finite': False,
        'grad_l2_norm': None,
        'lksa_gate_abs_max': None,
        'lksa_out_abs_max': None,
        'five_step_stable': False,
        'five_step_losses': [],
    }

    try:
        pred = net(x)
        loss = criterion(pred, y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('loss non-finite')
        result['forward_ok'] = True
        result['loss'] = float(loss.detach())
        result['lksa_gate_abs_max'] = float(max(gate_max_vals)) if gate_max_vals else None
        result['lksa_out_abs_max'] = float(max(lksa_out_max_vals)) if lksa_out_max_vals else None
        loss.backward()
        grads = [p.grad.detach() for p in net.parameters() if p.grad is not None]
        grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
        result['grad_finite'] = grad_finite
        if grad_finite:
            total_sq = sum(float(g.double().pow(2).sum()) for g in grads)
            result['grad_l2_norm'] = float(np.sqrt(total_sq))
    except Exception as exc:
        result['forward_error'] = repr(exc)
    finally:
        for h in handles:
            h.remove()

    if result['forward_ok'] and result['grad_finite']:
        # 在该失败现场做 5 步 Adam 试探更新（先重建与原训练同 lr 的优化器，验证不炸）
        net_step = build_network(opt).to(device)
        net_step.load_state_dict(case['params'], strict=True)
        net_step.train()
        optimizer = torch.optim.Adam(net_step.parameters(), lr=3.75e-4, betas=(0.9, 0.99))
        if 'optimizer' in case:
            try:
                optimizer.load_state_dict(case['optimizer'])
            except Exception:
                pass
        step_losses = []
        stable = True
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            try:
                out = net_step(x)
                loss_s = criterion(out, y)
                if not bool(torch.isfinite(loss_s)):
                    stable = False
                    break
                loss_s.backward()
                if not all(bool(torch.isfinite(p.grad).all()) for p in net_step.parameters() if p.grad is not None):
                    stable = False
                    break
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in net_step.parameters()):
                    stable = False
                    break
                step_losses.append(float(loss_s.detach()))
            except Exception:
                stable = False
                break
        result['five_step_stable'] = stable
        result['five_step_losses'] = step_losses

    return result


def inspect_normal_samples(
    ckpt_path: Path,
    net_opt: dict,
    split_file: Path,
    dataroot_lq: Path,
    dataroot_gt: Path,
    gammas: list[float | None],
    num_seqs: int,
    device: torch.device,
):
    """在正常验证序列上检查有界门控对 PSNR/SSIM 与 LKSA gate 分布的影响。"""
    names = [line.strip() for line in split_file.read_text(encoding='utf-8').splitlines() if line.strip()][:num_seqs]
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state = ckpt['params'] if 'params' in ckpt else ckpt
    psnr_opt = dict(type='calculate_psnr', crop_border=0, test_y_channel=False)
    ssim_opt = dict(type='calculate_ssim', crop_border=0, test_y_channel=False)

    summaries = []
    for gamma in gammas:
        net = build_network(dict(net_opt, lksa_gate_gamma=gamma)).to(device)
        net.load_state_dict(state, strict=True)
        net.eval()

        raw_gate_samples = []
        handles = []

        def make_hook(module: LKSA):
            def _hook(_mod, inputs, _output):
                inp = inputs[0].detach()
                yy = module.conv_big_v(
                    module.conv_big_h(module.conv_dv(module.conv_dh(module.conv_v(module.conv_h(inp)))))
                )
                raw_gate = (yy + module.conv3(inp)).abs().flatten()
                # 每层采样 4096 个点统计分布，避免内存膨胀
                step = max(1, raw_gate.numel() // 4096)
                raw_gate_samples.append(raw_gate[::step].cpu().numpy())

            return _hook

        for mod in net.modules():
            if isinstance(mod, LKSA):
                handles.append(mod.register_forward_hook(make_hook(mod)))

        psnrs, ssims = [], []
        try:
            with torch.no_grad():
                for name in names:
                    for t in range(9):
                        lq = cv2.imread(str(dataroot_lq / name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE)
                        gt = cv2.imread(str(dataroot_gt / name / f'{t:03d}.png'), cv2.IMREAD_GRAYSCALE)
                        inp = img2tensor(np.ascontiguousarray(lq[:, :, None]), bgr2rgb=False).div_(255.0).unsqueeze(0).to(device)
                        out = net(inp)
                        out_u8 = tensor2img([out])
                        psnrs.append(float(calculate_metric(dict(img=out_u8, img2=gt), psnr_opt)))
                        ssims.append(float(calculate_metric(dict(img=out_u8, img2=gt), ssim_opt)))
        finally:
            for h in handles:
                h.remove()

        all_gates = np.concatenate(raw_gate_samples) if raw_gate_samples else np.array([0.0])
        summaries.append({
            'gamma': gamma,
            'num_frames': len(psnrs),
            'psnr_mean': float(np.mean(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'raw_gate_abs_p50': float(np.percentile(all_gates, 50)),
            'raw_gate_abs_p95': float(np.percentile(all_gates, 95)),
            'raw_gate_abs_p99': float(np.percentile(all_gates, 99)),
            'raw_gate_abs_p999': float(np.percentile(all_gates, 99.9)),
            'raw_gate_abs_max': float(np.max(all_gates)),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', default='options/SAR/train_ucan_single_q2.yml')
    parser.add_argument('--normal-opt', default='options/SAR/test_ucan_single.yml')
    parser.add_argument('--normal-ckpt', default='experiments/ucan_sar_single/models/net_g_300000.pth')
    parser.add_argument('--split-file', default='docs/splits/sar_single_seed42/val.txt')
    parser.add_argument('--num-normal-seqs', type=int, default=20)
    parser.add_argument('--out-dir', default='docs/UCAN_exp1/diagnostics/lksa_gate')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    q2_opt = yaml_load(args.opt)
    norm_opt = yaml_load(args.normal_opt)
    ds_opt = next(iter(norm_opt['datasets'].values()))

    cases = [
        Path('docs/diagnostics/ucan_nonfinite/iter_129507_rank_0.pth'),
        Path('docs/diagnostics/ucan_nonfinite/iter_129507_rank_1.pth'),
    ]
    gammas = [None, 2.0, 4.0]

    failure_results = []
    for case_path in cases:
        for gamma in gammas:
            failure_results.append(
                inspect_failure_case(case_path, q2_opt['network_g'], q2_opt['train']['pixel_opt'], gamma, device)
            )
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    normal_results = inspect_normal_samples(
        Path(args.normal_ckpt),
        norm_opt['network_g'],
        Path(args.split_file),
        Path(ds_opt['dataroot_lq']),
        Path(ds_opt['dataroot_gt']),
        gammas,
        args.num_normal_seqs,
        device,
    )

    payload = {'failure_cases': failure_results, 'normal_samples': normal_results}
    (out_dir / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
