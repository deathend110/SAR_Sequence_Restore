"""生成配对实验的共享初始权重：保证同 seed 下 A/B/C/D1/D0 及第二轮变体拥有逐张量一致的初始参数。"""

import argparse
import json
from pathlib import Path

import torch

from basicsr.archs import build_network
from basicsr.utils import set_random_seed


def build_backbone_opt(lksa_gate_gamma: float | None = 4.0) -> dict:
    return {
        'type': 'UCAN',
        'upscale': 1,
        'in_chans': 1,
        'img_size': 128,
        'embed_dim': 48,
        'window_size': 16,
        'img_range': 1.0,
        'mlp_ratio': 1.0,
        'conv_depth': 5,
        'share': ['N', 'F', 'N', 'F'],
        'mhsa_num_heads': 2,
        'dfl_num_heads': 1,
        'use_checkpoint': False,
        'upsampler': None,
        'resi_connection': '1conv',
        'lksa_gate_gamma': lksa_gate_gamma,
    }


def build_group_network_opt(
    group: str,
    lksa_gate_gamma: float | None = 4.0,
    train_frame_chunk_size: int = 0,
    eval_frame_batch_size: int = 1,
) -> dict:
    bb_opt = build_backbone_opt(lksa_gate_gamma=lksa_gate_gamma)
    seq_extra = {
        'train_frame_chunk_size': train_frame_chunk_size,
        'eval_frame_batch_size': eval_frame_batch_size,
    }
    base_token = {
        'enabled': True,
        'in_ch': 48,
        'token_dim': 96,
        'num_heads': 4,
        'ffn_ratio': 2.0,
        'num_layers': 2,
        'num_frames': 9,
        'granularity': 'region',
        'tokenizer_type': 'grid',
        'mask_mode': 'full',
        'causal_lookahead': [1, 0],
        'use_frame_embed': True,
        'use_geom_bias': True,
        'use_ground_bias': True,
        'quality_mode': 'none',
        'quality_src_only': False,
        'writeback_mode': 'film',
        'use_delta_ln': False,
        'use_detail': False,
    }
    if group == 'A':
        return {'type': 'UCANSequence', 'backbone_opt': bb_opt, 'token_opt': {'enabled': False}, **seq_extra}
    if group == 'B':
        return {'type': 'UCANSequence', 'backbone_opt': bb_opt, 'token_opt': {**base_token, 'mask_mode': 'intra'}, **seq_extra}
    if group == 'C':
        return {'type': 'UCANSequence', 'backbone_opt': bb_opt, 'token_opt': {**base_token, 'mask_mode': 'full'}, **seq_extra}
    if group == 'D1':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'quality_mode': 'real'},
            **seq_extra,
        }
    if group == 'D0':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'quality_mode': 'constant'},
            **seq_extra,
        }
    if group == 'Causal_L1':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'causal', 'causal_lookahead': [1, 0]},
            **seq_extra,
        }
    if group == 'Causal_L0':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'causal', 'causal_lookahead': [0, 0]},
            **seq_extra,
        }
    if group == 'Causal_L2':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'causal', 'causal_lookahead': [2, 0]},
            **seq_extra,
        }
    if group == 'C_same_ground':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'same_ground'},
            **seq_extra,
        }
    if group == 'C_additive':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'writeback_mode': 'additive'},
            **seq_extra,
        }
    if group == 'C_detail':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'use_detail': True},
            **seq_extra,
        }
    if group == 'D1_detail':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'quality_mode': 'real', 'use_detail': True},
            **seq_extra,
        }
    if group == 'D1_src_only':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'quality_mode': 'real', 'quality_src_only': True},
            **seq_extra,
        }
    if group == 'D1_shuffle_q':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'mask_mode': 'full', 'quality_mode': 'shuffle'},
            **seq_extra,
        }
    if group == 'C_K1':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'granularity': 'global'},
            **seq_extra,
        }
    if group == 'C_K4':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'granularity': 'strip'},
            **seq_extra,
        }
    if group == 'C_hier':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'granularity': 'hierarchical'},
            **seq_extra,
        }
    if group == 'C_stats':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'tokenizer_type': 'stats'},
            **seq_extra,
        }
    if group == 'C_no_ground_bias':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'use_ground_bias': False},
            **seq_extra,
        }
    if group == 'C_no_frame_embed':
        return {
            'type': 'UCANSequence',
            'backbone_opt': bb_opt,
            'token_opt': {**base_token, 'use_frame_embed': False},
            **seq_extra,
        }
    raise ValueError(f'Unknown group: {group}')


def generate_paired_inits(
    backbone_ckpt: Path | None,
    seed: int,
    out_dir: Path,
    lksa_gate_gamma: float | None,
    groups: list[str],
) -> dict:
    """为同一 seed 的配对实验生成逐项共享初值的 checkpoint，并做张量级零差验证。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone_state = None
    if backbone_ckpt is not None:
        raw = torch.load(backbone_ckpt, map_location='cpu', weights_only=True)
        backbone_state = raw['params'] if 'params' in raw else raw

    # 先构建包含全部共同模块的基准参考模型：
    # 1) D1（含全部 K=16 token 模块与质量偏置模块）
    set_random_seed(seed)
    ref_d1 = build_network(build_group_network_opt('D1', lksa_gate_gamma=lksa_gate_gamma))
    if backbone_state is not None:
        ref_d1.load_backbone_state_dict(backbone_state, strict=True)
    ref_d1.token_interaction.reset_zero_init_branches()
    ref_d1_sd = ref_d1.state_dict()

    saved = {}
    for g in groups:
        # 使用同一 seed 创建各组特有参数（如 K!=16 或 detail/additive 等特有层），再逐项覆盖全部共同键
        set_random_seed(seed + 10007)
        net = build_network(build_group_network_opt(g, lksa_gate_gamma=lksa_gate_gamma))
        if backbone_state is not None:
            net.load_backbone_state_dict(backbone_state, strict=True)
        sd = net.state_dict()
        # 逐项复制与 ref_d1 完全同名的张量（包含 backbone、tokenizer、PE、两层 Transformer、wb_pre/dwconv/wb_out）
        copied_keys = 0
        for k, v in sd.items():
            if k in ref_d1_sd and ref_d1_sd[k].shape == v.shape:
                sd[k] = ref_d1_sd[k].clone()
                copied_keys += 1
        net.load_state_dict(sd, strict=True)
        if net.token_interaction is not None:
            net.token_interaction.reset_zero_init_branches()

        out_path = out_dir / f'init_{g}_s{seed}.pth'
        torch.save({'params': net.state_dict()}, out_path)
        saved[g] = {'path': str(out_path), 'copied_shared_keys': copied_keys, 'total_keys': len(sd)}

    # 配对严格校验：B/C 全部参数必须完全一致，D1/D0 全部参数必须完全一致，D1 包含 B/C 全部键值且差为 0
    if 'B' in saved and 'C' in saved:
        b_sd = torch.load(saved['B']['path'], map_location='cpu', weights_only=True)['params']
        c_sd = torch.load(saved['C']['path'], map_location='cpu', weights_only=True)['params']
        assert set(b_sd.keys()) == set(c_sd.keys())
        for k in b_sd:
            diff = (b_sd[k] - c_sd[k]).abs().max().item()
            assert diff == 0.0, f'B/C mismatch at {k}: {diff}'
    if 'D1' in saved and 'D0' in saved:
        d1_sd = torch.load(saved['D1']['path'], map_location='cpu', weights_only=True)['params']
        d0_sd = torch.load(saved['D0']['path'], map_location='cpu', weights_only=True)['params']
        assert set(d1_sd.keys()) == set(d0_sd.keys())
        for k in d1_sd:
            diff = (d1_sd[k] - d0_sd[k]).abs().max().item()
            assert diff == 0.0, f'D1/D0 mismatch at {k}: {diff}'
    if 'C' in saved and 'D1' in saved:
        c_sd = torch.load(saved['C']['path'], map_location='cpu', weights_only=True)['params']
        d1_sd = torch.load(saved['D1']['path'], map_location='cpu', weights_only=True)['params']
        for k in c_sd:
            diff = (c_sd[k] - d1_sd[k]).abs().max().item()
            assert diff == 0.0, f'C/D1 shared mismatch at {k}: {diff}'

    (out_dir / f'paired_init_manifest_s{seed}.json').write_text(
        json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', default=None, help='A_init 骨干权重路径（若省略则使用随机骨干）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--gamma', type=float, default=4.0)
    parser.add_argument(
        '--groups',
        nargs='+',
        default=['A', 'B', 'C', 'D1', 'D0'],
    )
    args = parser.parse_args()
    bb = Path(args.backbone) if args.backbone else None
    res = generate_paired_inits(bb, args.seed, Path(args.out_dir), args.gamma, args.groups)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
