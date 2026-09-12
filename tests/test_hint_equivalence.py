"""只比较 HINT 核心迁移，不加载任何预训练权重。"""
import os
from pathlib import Path
import types

import pytest
import torch

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


def test_hint_same_weights_and_inputs():
    root = Path(__file__).resolve().parents[1]
    reference = Path(os.environ.get('HINT_REFERENCE_PATH', str(root.parent / 'HINT/basicsr/models/archs/HINT_arch.py')))
    if not reference.is_file():
        pytest.skip('Set HINT_REFERENCE_PATH to the original HINT_arch.py for migration verification.')
    # 直接编译参考文件，避免导入 HINT fork 的同名 basicsr 包或写入来源目录缓存。
    module = types.ModuleType('hint_reference')
    exec(compile(reference.read_text(encoding='utf-8'), str(reference), 'exec'), module.__dict__)
    opt = yaml_load(str(root / 'options/SAR/train_hint_single.yml'))['network_g']
    torch.manual_seed(42)
    original = module.HINT(**{k: v for k, v in opt.items() if k != 'type'}).eval()
    migrated = build_network(opt).eval()
    migrated.load_state_dict(original.state_dict(), strict=True)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in migrated.parameters())
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for _ in range(2):
                x = torch.rand(1, 1, 32, 32)
                expected = original(x)
                actual = migrated(x)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(migrated.qv_cache, original.qv_cache, rtol=1e-5, atol=1e-6)
    finally:
        torch.set_num_threads(old_threads)
