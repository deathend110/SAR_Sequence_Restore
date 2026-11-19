"""生成按完整场景独立划分（A 方案）的数据列表与精确 H/L Manifest（指导书第 5.3、10.2 节）。"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

# 场景独立划分：train 5 场景、val 1 场景、test 1 场景
SCENE_SPLIT = {
    'train': ['bangkok', 'city2_histeq', 'filed', 'port', 'suburb'],
    'val': ['sar_figure'],
    'test': ['city1_histeq'],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_hlh_strip_profile():
    """依据 SAR Dataset Transformer/src/workflow/sarflow_build_hlh_mask.m 与 sarflow_frame_columns.m
    推导 1536 逻辑宽度的精确 0-based H 掩膜及 12 个 128px 条带覆盖比例。

    MATLAB 源码：
      logic_mask(1:512) = true; logic_mask(513:1024) = false; logic_mask(1025:1536) = true;
      full_mask = [repmat(true, 1, 344), logic_mask, repmat(true, 1, 344)]; % 1..2224
      t 帧 (1..9) columns = 1 + (t-1)*128 : (t-1)*128 + 1200;
      sar_image_roi_indices(1200 -> 600) 取 column_start = 1200/2 - 600/2 = 300 (1-based 300..899)；
      crop_center(600 -> 512) 取其中第 45..556 列，对应 1200 列中的 1-based 344..855。
      因此第 t 帧 (0..8) 局部列 c (0..511)（0-based 全局列 g = 128*t + c ∈ [0, 1535]）
      对应 2224 列 full_mask 的 1-based 列号为 344 + g：
        full_col ∈ [1..856]     <=> g ∈ [0, 512]     (0-based [0, 513))    -> H (True)
        full_col ∈ [857..1368]  <=> g ∈ [513, 1024]  (0-based [513, 1025)) -> L (False)
        full_col ∈ [1369..2224] <=> g ∈ [1025, 1535] (0-based [1025, 1536))-> H (True)
    边界恰为指导书第 5.3 节提示核对的 x=513, 1025（含 1 像素 ROI 起始偏置）。
    """
    logic_mask_h = [False] * 1536
    for g in range(513):
        logic_mask_h[g] = True
    for g in range(1025, 1536):
        logic_mask_h[g] = True

    strip_q = []
    for s in range(12):
        seg = logic_mask_h[s * 128:(s + 1) * 128]
        strip_q.append(sum(seg) / 128.0)

    return {
        'logic_length': 1536,
        'valid_margin': 344,
        'patch_size': 512,
        'step': 128,
        'n_frames': 9,
        'h_intervals_0based_inclusive_exclusive': [[0, 513], [1025, 1536]],
        'l_intervals_0based_inclusive_exclusive': [[513, 1025]],
        'h_intervals_1based_inclusive': [[1, 513], [1026, 1536]],
        'l_intervals_1based_inclusive': [[514, 1025]],
        'strip_quality_s0_to_s11': strip_q,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        default='/data/cjz/SAR_Transformer/dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64',
    )
    parser.add_argument('--out-dir', default='docs/UCAN_exp1/splits/scene_split_v1')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    lq_root = data_root / 'LQ'
    gt_root = data_root / 'GT'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lq_seqs = {p.name for p in lq_root.iterdir() if p.is_dir()}
    gt_seqs = {p.name for p in gt_root.iterdir() if p.is_dir()}
    assert lq_seqs == gt_seqs, 'LQ 与 GT 序列集合不一致'
    all_seqs = sorted(lq_seqs)

    by_scene = defaultdict(list)
    for name in all_seqs:
        scene = name.split('_rstart_')[0]
        for t in range(9):
            fname = f'{t:03d}.png'
            assert (lq_root / name / fname).is_file(), f'Missing {lq_root / name / fname}'
            assert (gt_root / name / fname).is_file(), f'Missing {gt_root / name / fname}'
        by_scene[scene].append(name)

    expected_scenes = set(SCENE_SPLIT['train'] + SCENE_SPLIT['val'] + SCENE_SPLIT['test'])
    assert set(by_scene.keys()) == expected_scenes, f'场景集合不匹配: {set(by_scene.keys())} vs {expected_scenes}'

    split_stats = {}
    hashes = {}
    seen = set()
    for split_name, scenes in SCENE_SPLIT.items():
        seq_list = []
        for s in sorted(scenes):
            seq_list.extend(sorted(by_scene[s]))
        seq_list = sorted(seq_list)
        assert not (seen & set(seq_list)), f'划分存在重叠: {split_name}'
        seen.update(seq_list)

        out_file = out_dir / f'{split_name}.txt'
        out_file.write_text('\n'.join(seq_list) + '\n', encoding='utf-8')
        hashes[f'{split_name}.txt'] = sha256_file(out_file)
        split_stats[split_name] = {
            'scenes': {s: len(by_scene[s]) for s in scenes},
            'num_sequences': len(seq_list),
            'num_frames': len(seq_list) * 9,
            'ratio': len(seq_list) / len(all_seqs),
        }

    assert seen == set(all_seqs), '划分未覆盖全部序列'

    manifest = {
        'split_scheme': 'Scheme A: Scene-disjoint independent split (v1)',
        'data_root': str(data_root),
        'total_sequences': len(all_seqs),
        'total_frames': len(all_seqs) * 9,
        'splits': split_stats,
        'sha256': hashes,
        'hlh_geometry': build_hlh_strip_profile(),
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest['sha256']['manifest.json'] = sha256_file(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
