"""训练结束后自动清理中间检查点，只保留最优 PSNR、最优 SSIM 和最后一次（latest/last）。"""

import argparse
import json
import os
import re
from pathlib import Path


def _parse_val_history(exp_dir: Path) -> list[dict]:
    """优先读取 val_history.json；若为单帧训练则解析最新 train_*.log。"""
    hist_file = exp_dir / 'val_history.json'
    if hist_file.is_file():
        data = json.loads(hist_file.read_text(encoding='utf-8'))
        records = []
        for entry in data:
            metrics = entry.get('metrics', {})
            records.append({
                'iter': int(entry['iter']),
                'psnr': float(metrics.get('psnr', float('-inf'))),
                'ssim': float(metrics.get('ssim', float('-inf'))),
            })
        return records

    # 回退：从最新的 train_*.log 中直接解析每个 Validation 区块并优先使用最后一个区块内记录的 Best @ iter
    logs = sorted(exp_dir.glob('train_*.log'), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f'Neither val_history.json nor train_*.log found in {exp_dir}')
    text = logs[-1].read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    records = []
    last_iter = None
    best_psnr_from_log = None
    best_ssim_from_log = None
    i = 0
    while i < len(lines):
        m_iter = re.search(r'iter:\s*(\d[\d,]*)', lines[i])
        if m_iter:
            last_iter = int(m_iter.group(1).replace(',', ''))
        if re.search(r'Validation\s+\S+', lines[i]) and i + 2 < len(lines):
            m_p = re.search(r'#\s*psnr:\s*([0-9.+-]+)\s+Best:\s*([0-9.+-]+)\s*@\s*(\d+)\s*iter', lines[i + 1])
            m_s = re.search(r'#\s*ssim:\s*([0-9.+-]+)\s+Best:\s*([0-9.+-]+)\s*@\s*(\d+)\s*iter', lines[i + 2])
            if m_p and m_s:
                best_psnr_from_log = (int(m_p.group(3)), float(m_p.group(2)))
                best_ssim_from_log = (int(m_s.group(3)), float(m_s.group(2)))
                if last_iter is not None:
                    records.append({
                        'iter': int(last_iter),
                        'psnr': float(m_p.group(1)),
                        'ssim': float(m_s.group(1)),
                    })
                i += 3
                continue
        i += 1

    if best_psnr_from_log is not None and best_ssim_from_log is not None:
        records.append({'iter': best_psnr_from_log[0], 'psnr': best_psnr_from_log[1], 'ssim': float('-inf')})
        records.append({'iter': best_ssim_from_log[0], 'psnr': float('-inf'), 'ssim': best_ssim_from_log[1]})
    return records


def prune_experiment(exp_dir: Path, dry_run: bool = False) -> dict:
    """只保留最优 PSNR、最优 SSIM 和 last/latest 检查点与对应 training_state，其余一律删除。"""
    models_dir = exp_dir / 'models'
    states_dir = exp_dir / 'training_states'
    if not models_dir.is_dir():
        raise FileNotFoundError(f'Models directory not found: {models_dir}')

    # 扫描现有整数 iter 的 .pth 文件
    iter_to_pth = {}
    for p in models_dir.iterdir():
        if p.is_symlink() or not p.is_file():
            continue
        m = re.fullmatch(r'net_g_(\d+)\.pth', p.name)
        if m:
            iter_to_pth[int(m.group(1))] = p

    if not iter_to_pth:
        raise FileNotFoundError(f'No numbered net_g_<iter>.pth found in {models_dir}')

    records = _parse_val_history(exp_dir)
    # 仅在确实保存了对应 iter 模型文件的验证记录中选择最优
    saved_records = [r for r in records if r['iter'] in iter_to_pth]
    if not saved_records:
        # 若训练结束时最新模型保存为 net_g_latest.pth 且最后一次验证在 total_iter 但没有对应编号 pth，
        # 回退匹配最接近的保存点，或允许 latest 对应最大 iter
        saved_records = records

    last_iter = max(iter_to_pth.keys())
    best_psnr_rec = max(saved_records, key=lambda r: (r['psnr'], r['iter']))
    best_ssim_rec = max(saved_records, key=lambda r: (r['ssim'], r['iter']))

    best_psnr_iter = best_psnr_rec['iter'] if best_psnr_rec['iter'] in iter_to_pth else last_iter
    best_ssim_iter = best_ssim_rec['iter'] if best_ssim_rec['iter'] in iter_to_pth else last_iter

    keep_iters = {best_psnr_iter, best_ssim_iter, last_iter}
    latest_path = models_dir / 'net_g_latest.pth'

    removed_models = []
    removed_states = []
    bytes_freed = 0

    for p in sorted(models_dir.iterdir()):
        if p.name in ('net_g_best_psnr.pth', 'net_g_best_ssim.pth'):
            continue
        if p.name == 'net_g_latest.pth':
            continue
        m = re.fullmatch(r'net_g_(\d+)\.pth', p.name)
        if m and int(m.group(1)) in keep_iters:
            continue
        if p.is_file() or p.is_symlink():
            bytes_freed += p.stat().st_size if p.is_file() and not p.is_symlink() else 0
            removed_models.append(p.name)
            if not dry_run:
                p.unlink()

    if states_dir.is_dir():
        for p in sorted(states_dir.iterdir()):
            m = re.fullmatch(r'(\d+)\.state', p.name)
            if m and int(m.group(1)) in keep_iters:
                continue
            if p.is_file() or p.is_symlink():
                bytes_freed += p.stat().st_size if p.is_file() and not p.is_symlink() else 0
                removed_states.append(p.name)
                if not dry_run:
                    p.unlink()

    # 创建便于下游评估引用的符号链接 net_g_best_psnr.pth / net_g_best_ssim.pth
    if not dry_run:
        for alias, target_iter in (
            ('net_g_best_psnr.pth', best_psnr_iter),
            ('net_g_best_ssim.pth', best_ssim_iter),
        ):
            alias_path = models_dir / alias
            if alias_path.exists() or alias_path.is_symlink():
                alias_path.unlink()
            os.symlink(f'net_g_{target_iter}.pth', alias_path)

    manifest = {
        'exp_dir': str(exp_dir),
        'dry_run': dry_run,
        'best_psnr': {'iter': best_psnr_iter, 'val': best_psnr_rec['psnr']},
        'best_ssim': {'iter': best_ssim_iter, 'val': best_ssim_rec['ssim']},
        'last_iter': last_iter,
        'has_latest_file': latest_path.exists(),
        'kept_iters': sorted(keep_iters),
        'removed_models': removed_models,
        'removed_states': removed_states,
        'bytes_freed': bytes_freed,
    }
    if not dry_run:
        (exp_dir / 'checkpoint_manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-dir', required=True, help='实验输出目录，如 experiments/ucan_seq_C_s42')
    parser.add_argument('--dry-run', action='store_true', help='仅打印将删除的文件，不实际删除')
    args = parser.parse_args()
    manifest = prune_experiment(Path(args.exp_dir), dry_run=args.dry_run)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
