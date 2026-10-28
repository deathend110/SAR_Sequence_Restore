# SAR 双卡短程验证（2026-10-28）

本实验用于筛选训练加速方案，不运行完整训练。TODO 与执行结果见本目录及最终报告。

## 复现

在仓库根目录运行：

```bash
.venv/bin/python docs/experiments/sar_short_20261028/run_all.py
.venv/bin/python docs/experiments/sar_short_20261028/summarize.py
```

脚本固定 GPU 0、1，按顺序启动独立的双进程 DDP 作业，单个训练作业最多 400 步。
每个作业最长 10 分钟，整体运行预算 40 分钟；超时结束对应进程组。
结果默认写入 `experiments/sar_short_20261028/`，复现会覆盖同名实验结果，需先另行保存旧结果。汇总脚本核查指标和摘要后生成 `docs/SAR_short_validation_report_20261028.md`。

## 固定条件

- 数据集：`range_2d_sft_q2_s0_fr1p3_fa0p4_phi0_lp0p99_hp99p9_roi600`。
- 模型与裁剪：沿用正式 Q=2 配置，训练裁剪 128×128，评估完整 512×512。
- 精度：默认 FP32，不启用 AMP，不额外修改 TF32；实际后端设置写入各作业日志。
- 训练：完整 seed42 train 划分，沿用 EnlargedSampler 和随机裁剪；每个 rank 的 seed 为 42+rank。
- 评估：val/test 各按场景排序，从每组排序后的序列中以独立 seed42 随机选择一条，共七条序列、63 帧。
- 指标：逐帧 uint8 image-based PSNR/SSIM，crop_border=0，不转换 Y 通道，全帧等权平均。
- 分布式评估：固定 `range(rank, N, 2)` 分片，汇总实际帧记录；HINT 从冷缓存开始，分片内连续更新缓存。
- 每次评估恢复 HINT 原训练缓存，使中途评估不改变后续训练轨迹；本约定与正式训练保留验证后缓存的行为有区别。
- 微调从历史最佳 PSNR `.pth` 初始化，重新创建优化器，不加载旧 `.state`。
- 阶段：每 1/4 训练预算做验证，以验证 PSNR 选择权重，将训练前的 step0 一并纳入候选，最后仅评估一次选定权重的测试子集。
- 两种微调各处理 2400 个样本：SwinIR 400×3×2，HINT 200×6×2。
- Batch 测速：各独立进程运行 50 步，排除前 10 步，计时包含真实取数据、前后向、优化器与 DDP 同步。
- Warmup 对照：HINT 同种子、同数据、同初始权重，从头训练 400 步；分别禁用 warmup、设为 3000 步，观察启动阶段。为稳定对照，单独关闭 TF32、禁用 cuDNN benchmark、设置 cuDNN deterministic 和确定性算法请求（不支持时记录警告），并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。首步保存初值与输入摘要。

原数据集与新数据集除量化配置外，`fr`、`fa` 等参数也发生变化；直接评估反映两套数据之间的迁移表现。
场景均衡子集与完整数据的场景占比不同，其均值用于本次配对对照。

## 文件

- `TODO.md`：任务进度。
- `run_check.py`：单个实验入口，含固定样本选择和指标计算。
- `run_all.py`：顺序调度、运行时间限制和 GPU 监控。
- `experiments/sar_short_20261028/*.json`：实验汇总、逐步 loss、配置和作业记录。
- `experiments/sar_short_20261028/metrics/`：逐帧指标及按场景、帧位置汇总。
- `experiments/sar_short_20261028/models/`：独立保存的短程最佳权重。
- `experiments/sar_short_20261028/gpu_telemetry.csv`：每 5 秒记录两张 GPU 的利用率、显存、功耗和温度。
- 本轮另有 `baseline_hint_repeat`、`audit_*` 和未固定后端的 `warmup_hint_*` 复核/探索记录。正式 warmup 对照采用 `controlled_warmup_hint_*`；探索记录不作为 warmup 因果结论的依据。
