"""核对实验输出，生成中文报告与可检查的汇总证据。"""

import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'experiments/sar_short_20261028'
HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / f'{name}.json').read_text(encoding='utf-8'))


def main():
    baselines = {model: load(f'baseline_{model}') for model in ('swinir', 'hint')}
    fine = {model: load(f'finetune_{model}') for model in baselines}
    batches = {model: load(f'batch_{model}_b{batch}') for model, batch in [('swinir', 3), ('hint', 6)]}
    warm = {str(w): load(f'controlled_warmup_hint_w{w}') for w in (-1, 3000)}
    jobs = load('jobs')
    provenance = load('provenance')
    repeat = load('baseline_hint_repeat')
    for path, expected in provenance['checkpoint_sha256'].items():
        with (ROOT / path).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    for job in jobs:
        if job['returncode']:
            assert job['name'].startswith('batch_')
            assert 'out of memory' in (OUT / f'{job["name"]}.log').read_text(encoding='utf-8').lower()
    # 从逐帧记录重新计算均值，检查固定划分、帧数和全部指标。
    metric_files = list((OUT / 'metrics').glob('*.json'))
    for path in metric_files:
        obj = json.loads(path.read_text(encoding='utf-8'))
        rows, summary = obj['rows'], obj['summary']
        assert len(rows) == len({(r['sequence'], r['frame']) for r in rows}) == 63
        assert set(r['sequence'] for r in rows) == set(provenance['subsets'][summary['split']])
        for key in ('input_psnr', 'input_ssim', 'psnr', 'ssim'):
            assert all(math.isfinite(row[key]) for row in rows)
            assert abs(statistics.mean(row[key] for row in rows) - summary[key]) < 1e-10
    for result in [*fine.values(), *batches.values(), *warm.values()]:
        assert len(result['records']) == result['steps'] <= 400
        assert all(math.isfinite(r['loss']) for r in result['records'])
    assert warm['-1']['records'][0]['loss'] == warm['3000']['records'][0]['loss']
    assert load('controlled_warmup_hint_w-1_digests') == load('controlled_warmup_hint_w3000_digests')
    for model, result in fine.items():
        winner = max(result['evaluations'], key=lambda e: e['psnr'])
        assert winner['tag'].endswith(f'_step{result["best_psnr_step"]}')
        assert Path(result['checkpoint']).is_file()
    snapshots = {}
    for name in [
            'baseline_swinir', 'baseline_hint', 'baseline_hint_repeat', 'finetune_swinir', 'finetune_hint',
            'batch_swinir_b3', 'batch_hint_b6', 'warmup_hint_w-1', 'warmup_hint_w3000', 'controlled_warmup_hint_w-1',
            'controlled_warmup_hint_w3000'
    ]:
        value = load(name)
        value.pop('records', None)
        snapshots[name] = value
    audits = {
        name: load(name)
        for name in ('audit_warmup_hint_w-1', 'audit_warmup_hint_w3000', 'audit_deterministic_hint_w-1',
                     'audit_deterministic_hint_w3000')
    }
    evidence = dict(
        provenance=provenance,
        jobs=jobs,
        results=snapshots,
        audits=audits,
        warmup_digests=load('controlled_warmup_hint_w-1_digests'),
        verification=dict(
            metric_files=len(metric_files),
            frames_per_evaluation=63,
            original_checkpoints_unchanged=True,
            controlled_first_loss_equal=True))
    (HERE / 'results_summary.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    ends = [datetime.datetime.fromisoformat(job['ended']) for job in jobs]
    starts = [end - datetime.timedelta(seconds=job['seconds']) for end, job in zip(ends, jobs)]
    minutes = (max(ends) - min(starts)).total_seconds() / 60
    failed = sum(job['returncode'] != 0 for job in jobs)
    lines = [
        '# SAR 双卡短程验证报告（2026-10-28）',
        '',
        '## 结论与建议',
        '',
        '现有 HINT 权重可作为 Q=2 的快速恢复起点。本轮微调没有带来明确的整体收益；FP32 下增大 batch 的测试均触及显存上限。建议保留 SwinIR 每卡 3、HINT 每卡 6，以及此前缩短的正式训练预算。',
        '',
        '## 执行范围',
        '',
        f'- 使用 GPU 0、1，两张 RTX 4090；PyTorch 2.6.0+cu124、BasicSR 1.4.2。',
        f'- 作业时段：{min(starts):%H:%M:%S}～{max(ends):%H:%M:%S}（Asia/Shanghai），约 {minutes:.1f} 分钟。',
        f'- 共 {len(jobs)} 个独立作业，其中 {failed} 个为触及容量边界的 OOM 测试；其他作业正常完成。单个训练作业最多 400 步。',
        '- 不启用 AMP；基线、微调和 batch 测速沿用默认后端设置（matmul TF32=false、cuDNN TF32=true）。',
        '- 生产训练代码、正式 YML、历史权重均未改动；新增文件仅为实验脚本、TODO 和报告。',
        '- 验证/测试各选七类场景各一条序列、全部 F0～F8，共各 63 帧；训练仍使用完整 train 划分。',
        '- 指标沿用 uint8 image-based PSNR/SSIM、crop_border=0、非 Y 通道；评估完整 512×512，训练随机裁剪 128×128。',
        '- HINT 评估使用固定双卡分片和冷缓存，评估后恢复训练缓存，保证中途评估不改变训练轨迹。',
        '',
        '## 1. 已有权重直接评估 Q=2',
        '',
        '| 模型 | 历史权重迭代 | 验证 PSNR / SSIM | 测试 PSNR / SSIM |',
        '|---|---:|---:|---:|',
    ]
    for model, iteration in [('swinir', 493358), ('hint', 813496)]:
        val, test = baselines[model]['evaluations']
        lines.append(
            f'| {model.upper()} | {iteration} | {val["psnr"]:.4f} / {val["ssim"]:.4f} | {test["psnr"]:.4f} / {test["ssim"]:.4f} |'
        )
    inp = baselines['swinir']['evaluations'][1]
    lines += [
        '', f'测试子集输入为 {inp["input_psnr"]:.4f} dB / {inp["input_ssim"]:.4f}。这里比较的是各自已有权重的可用性，两份历史权重的训练曝光量不同。', '',
        '## 2. 短程微调与阶段检查', '',
        '加载历史最优 PSNR `.pth`，新建 Adam，学习率 5e-5，在预算 75% 处减半；不加载旧 `.state`。SwinIR 400 步、HINT 200 步，各处理 2400 个训练样本。每 1/4 预算验证一次，以验证 PSNR 选权重，step0 也参与选择；选定后评估测试子集。',
        '', '| 模型 | 验证选中步数 | 测试 PSNR / SSIM | 相对原权重 ΔPSNR / ΔSSIM |', '|---|---:|---:|---:|'
    ]
    for model, result in fine.items():
        old = baselines[model]['evaluations'][1]
        new = result['test']
        lines.append(
            f'| {model.upper()} | {result["best_psnr_step"]} | {new["psnr"]:.4f} / {new["ssim"]:.4f} | {new["psnr"]-old["psnr"]:+.4f} / {new["ssim"]-old["ssim"]:+.4f} |'
        )
    lines += [
        '', 'SwinIR 的 PSNR 增益很小，七个场景中四个上升、三个下降；HINT 测试 PSNR 基本持平、SSIM 略降。本轮结果支持先使用已有权重，不支持直接投入长时间微调。', '',
        '| 模型 | 步数 | 验证 PSNR | 验证 SSIM |', '|---|---:|---:|---:|'
    ]
    for model, result in fine.items():
        for value in result['evaluations']:
            step = value['tag'].rsplit('step', 1)[1]
            lines.append(f'| {model.upper()} | {step} | {value["psnr"]:.4f} | {value["ssim"]:.4f} |')
    lines += [
        '', '## 3. Batch 容量与吞吐量', '',
        '各档独立启动双卡 DDP，计划运行 50 步，排除前 10 步；稳态计时包含取数据、前后向、优化器和同步。峰值为两卡中较大的 PyTorch allocated 显存，和 nvidia-smi 进程占用口径不同。',
        '', '| 模型 | 每卡 batch | 状态 | 样本/秒（双卡） | 峰值 allocated GiB/卡 |', '|---|---:|---|---:|---:|'
    ]
    for model, sizes in [('swinir', (3, 4, 6)), ('hint', (6, 8, 10))]:
        for size in sizes:
            result = batches[model]
            if size == result['batch_per_gpu']:
                lines.append(
                    f'| {model.upper()} | {size} | 完成 | {result["samples_per_second"]:.2f} | {result["peak_allocated_gib"]:.2f} |'
                )
            else:
                lines.append(f'| {model.upper()} | {size} | CUDA OOM | — | — |')
    lines += [
        '', '维持当前 batch。训练取数据平均耗时约 0.002～0.003 秒，显著小于计算时间，暂不优先增加 worker 或改造数据存储。', '', '## 4. HINT warmup 启动对照', '',
        '先固定种子执行探索对照，复核发现默认后端下首步 loss 不一致。单步审计确认权重及两卡首批输入 SHA256 一致；固定后端后两组首步 loss 均为 0.3648087978。', '',
        '因此以下使用重新运行的受控对照：禁用 cuDNN benchmark、关闭 TF32、启用 cuDNN deterministic 和确定性算法请求，并设置 CUBLAS_WORKSPACE_CONFIG=:4096:8。两组仍各 400 步，初始权重和首批输入摘要相同，种子及训练数据顺序固定。',
        '', '| 步数 | 无 warmup PSNR / SSIM | 3000 步 warmup PSNR / SSIM |', '|---:|---:|---:|'
    ]
    for a, b in zip(warm['-1']['evaluations'], warm['3000']['evaluations']):
        step = a['tag'].rsplit('step', 1)[1]
        lines.append(f'| {step} | {a["psnr"]:.4f} / {a["ssim"]:.4f} | {b["psnr"]:.4f} / {b["ssim"]:.4f} |')
    loss_a = statistics.mean(r['loss'] for r in warm['-1']['records'][-100:])
    loss_b = statistics.mean(r['loss'] for r in warm['3000']['records'][-100:])
    lines += [
        '', f'末 100 步平均 Charbonnier loss：无 warmup {loss_a:.6f}，3000 步 warmup {loss_b:.6f}。两组均完成且 loss 有限。', '',
        f'第 400 步 warmup 学习率为 {warm["3000"]["records"][-1]["lr"]:.8g}，无 warmup 为 0.0003464。该对照观察到 3000 步预热的起步更慢，当前正式配置保留 warmup_iter=-1。其最终收敛效果由完整预热后的实验评价。',
        '', '固定后端是对照质量控制措施，不作为提速方案；此处的运行速度不与默认后端的 batch 测速直接混合比较。', '', '## 5. 重复性与结果适用范围', '',
        f'HINT 原权重测试 PSNR 两次为 {baselines["hint"]["evaluations"][1]["psnr"]:.6f} 和 {repeat["evaluations"][1]["psnr"]:.6f}，变化约 {abs(baselines["hint"]["evaluations"][1]["psnr"]-repeat["evaluations"][1]["psnr"]):.6f} dB。',
        '',
        '本轮以场景均衡的 63 帧子集和有限训练预算筛选方案。微调是否推广，应结合完整验证集及各场景收益继续判断。原、新数据集的 fr、fa 等参数也不同，已有权重直接评估反映整套数据设置间的迁移表现。HINT 的本次缓存隔离协议与正式训练验证后继续保留缓存的行为有区别。',
        '', '## 6. 推荐落地设置与文件', '', '- 保留已更新的正式预算：SwinIR 80 万步、HINT 40 万步及对应提前衰减节点。',
        '- 保留每卡 batch：SwinIR 3、HINT 6；当前 FP32 模型不直接增大 batch。',
        '- Q=2 快速恢复优先试已有 HINT 权重；SwinIR 短程微调权重保留为候选，HINT 优先保留原权重。', '- HINT 当前保留无 warmup；涉及小幅数值差异的启动实验先固定后端并核查首步。',
        '- 阶段选择优先验证集最优权重，避免仅使用最后一组。', '', '- [已完成 TODO](experiments/sar_short_20261028/TODO.md)',
        '- [汇总证据和输入摘要](experiments/sar_short_20261028/results_summary.json)',
        '- [复现说明](experiments/sar_short_20261028/README.md)', '- [原始结果目录](../experiments/sar_short_20261028/)',
        '- [SwinIR 微调候选权重](../experiments/sar_short_20261028/models/finetune_swinir_best.pth)',
        '- [HINT 微调候选权重](../experiments/sar_short_20261028/models/finetune_hint_best.pth)', '',
        f'已从 {len(metric_files)} 份逐帧结果重新计算均值，核对帧数、唯一性、划分归属、有限指标、权重选择及历史权重摘要。'
    ]
    (ROOT / 'docs/SAR_short_validation_report_20261028.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    todo = HERE / 'TODO.md'
    todo.write_text(todo.read_text(encoding='utf-8').replace('- [ ]', '- [x]'), encoding='utf-8')
    print(f'报告与汇总证据已生成；核对 {len(metric_files)} 份指标文件、{len(jobs)} 个作业。')


if __name__ == '__main__':
    main()
