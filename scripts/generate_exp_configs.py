"""批量生成 UCAN Stage 3 / Stage 4 配对实验 YAML 配置文件。"""

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.init_paired_weights import build_group_network_opt

DEFAULT_DATAROOT_LQ = '/data/cjz/SAR_Transformer/dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64/LQ'
DEFAULT_DATAROOT_GT = '/data/cjz/SAR_Transformer/dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64/GT'
DEFAULT_VAL_ITERS = [4000, 8000, 12000, 14000, 16000, 18000, 20000]

ROUND1_GROUPS = ['A', 'B', 'C', 'D1', 'D0']
ROUND2_GROUPS = [
    'C_K1',
    'C_K4',
    'C_hier',
    'C_stats',
    'D1_src_only',
    'D1_shuffle_q',
    'C_additive',
    'D1_additive',
    'C_detail',
    'D1_detail',
    'C_no_ground_bias',
    'C_no_frame_embed',
    'C_same_ground',
    'Causal_L1',
    'Causal_L0',
    'Causal_L2',
]


def build_sequence_train_config(
    group: str,
    seed: int,
    init_dir: str = 'experiments/paired_inits_v1',
    val_iters: list[int] | None = None,
    lksa_gate_gamma: float = 4.0,
    train_frame_chunk_size: int = 3,
    eval_frame_batch_size: int = 1,
) -> dict:
    """构建单组跨帧 Token 训练与评估配置字典（严格配对采样与统一评估口径）。"""
    iters = list(val_iters) if val_iters is not None else list(DEFAULT_VAL_ITERS)
    run_name = f'ucan_seq_{group}_s{seed}'
    init_ckpt = f'{init_dir}/s{seed}/init_{group}_s{seed}.pth'

    return {
        'name': run_name,
        'model_type': 'UCANSequenceModel',
        'scale': 1,
        'num_gpu': 2,
        'manual_seed': int(seed),
        'datasets': {
            'train': {
                'name': 'sar_seq_train',
                'type': 'SARSequenceDataset',
                'dataroot_lq': DEFAULT_DATAROOT_LQ,
                'dataroot_gt': DEFAULT_DATAROOT_GT,
                'split_file': 'docs/UCAN_exp1/splits/scene_split_v1/train.txt',
                'crop_height': 64,
                'deterministic_crop_seed': int(seed),
                'grid_h': 4,
                'grid_w': 4,
                'use_shuffle': True,
                'num_worker_per_gpu': 4,
                'batch_size_per_gpu': 1,
                'dataset_enlarge_ratio': 1,
                'prefetch_mode': None,
            },
            'val': {
                'name': 'sar_seq_val',
                'type': 'SARSequenceDataset',
                'dataroot_lq': DEFAULT_DATAROOT_LQ,
                'dataroot_gt': DEFAULT_DATAROOT_GT,
                'split_file': 'docs/UCAN_exp1/splits/scene_split_v1/val.txt',
                'crop_height': None,
                'grid_h': 4,
                'grid_w': 4,
            },
        },
        'network_g': build_group_network_opt(
            group,
            lksa_gate_gamma=lksa_gate_gamma,
            train_frame_chunk_size=train_frame_chunk_size,
            eval_frame_batch_size=eval_frame_batch_size,
        ),
        'path': {
            'pretrain_network_g': init_ckpt,
            'strict_load_g': True,
            'param_key_g': 'params',
            'resume_state': None,
        },
        'train': {
            'total_iter': 20000,
            'warmup_iter': -1,
            'ema_decay': 0,
            'optim_g': {
                'type': 'Adam',
                'lr': 1e-4,
                'weight_decay': 0,
                'betas': [0.9, 0.99],
            },
            'scheduler': {
                'type': 'MultiStepLR',
                'milestones': [12000, 17000],
                'gamma': 0.5,
            },
            'pixel_opt': {
                'type': 'CharbonnierLoss',
                'loss_weight': 1.0,
                'reduction': 'mean',
                'eps': 1e-9,
            },
        },
        'val': {
            'val_freq': 20000,
            'val_iters': iters,
            'save_img': False,
            'pbar': False,
            'metrics': {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': False,
                },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': False,
                },
            },
        },
        'logger': {
            'print_freq': 200,
            'save_checkpoint_freq': 20000,
            'save_checkpoint_iters': iters,
            'use_tb_logger': True,
            'wandb': {
                'project': None,
                'resume_id': None,
            },
        },
    }


def write_exp_configs(
    out_dir: Path,
    groups: list[str],
    seeds: list[int],
    init_dir: str = 'experiments/paired_inits_v1',
) -> list[Path]:
    """将指定 group × seed 列表生成到目标配置目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        for group in groups:
            cfg = build_sequence_train_config(group=group, seed=seed, init_dir=init_dir)
            out_path = out_dir / f'{group}_s{seed}.yml'
            text = (
                f'# 自动生成的配对实验配置：{group} (seed={seed})，严禁手工偏离共享默认值\n'
                + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            )
            out_path.write_text(text, encoding='utf-8')
            written.append(out_path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='options/SAR/cross_frame_v1')
    parser.add_argument('--groups', nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--init-dir', default='experiments/paired_inits_v1')
    args = parser.parse_args()

    paths = write_exp_configs(Path(args.out_dir), args.groups, args.seeds, init_dir=args.init_dir)
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()
