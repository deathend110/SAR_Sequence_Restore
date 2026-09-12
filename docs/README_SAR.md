# SAR_Sequence_Restore 实验使用说明

本工作区以标准 BasicSR 1.4.2 为主框架，提供单比特 SAR 的 SwinIR、HINT 单帧基线。九帧按独立样本训练，保留 sequence 与 F₀～F₈ 信息用于评估。问题定义见 [Define.md](Define.md)，迁移来源与验证见 [迁移记录](migration_record.md)。

## 环境

以下所有命令均从新工作区根目录执行：

```bash
cd /data/cjz/SAR_Sequence_Restore
uv sync --locked --extra sar
uv run --no-sync python -c 'import basicsr, torch; print(basicsr.__file__); print(torch.__version__)'
```

锁定 Python 3.12、PyTorch 2.6.0+cu124、torchvision 0.21.0、NumPy 1.26.4。BasicSR 以 editable 形式安装到本目录；SwinIR/HINT 不需要编译 BasicSR 的可选 CUDA 扩展。`sar` extra 提供 einops、matplotlib；开发组提供 pytest 及迁移参考测试依赖。

## 数据和训练协议

数据仍位于原仓库的 `dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64/{LQ,GT}`。YAML 中配置绝对数据路径，split 引用新工作区的 `docs/splits/sar_single_seed42/`。迁移机器时修改两种模型的训练、测试 YAML 中数据路径。

| 项目 | 设置 |
| --- | --- |
| 划分 | seed 42，sequence 级 9:1，同一序列九帧同属一组 |
| 训练 | 3,688 条 sequence，33,192 帧 |
| 验证／测试 | 410 条 sequence，3,690 帧；两者使用同一列表 |
| 图像 | 单通道、[0,1]；训练共同裁剪 128×128，不旋转翻转；验证整图 512×512 |
| 多卡 | 官方 BasicSR DistributedDataParallel（DDP）；通过 `torchrun` + `--launcher pytorch` 启动；world_size=2；物理 GPU 0、1 |
| Batch／workers | `batch_size_per_gpu: 3`；每个 rank batch=3，global batch=3×2=6；`num_worker_per_gpu: 4`；两个 rank 合计 workers=8 |
| 种子／数据重复倍率 | 42／1 |
| 损失／优化器 | Charbonnier eps=1e-9，mean；Adam lr=1.732e-4、betas=[0.9,0.999]、weight decay=0 |
| 调度 | MultiStepLR，milestones=[1066667,1600000,1866667,2000000,2133333]，gamma=0.5 |
| 总步数 | 2,200,000，到达预算后按官方入口结束 |
| 日志／保存／验证 | 每 267／6667／6667 步 |
| EMA／warmup | 关闭 |

两份配置除网络、官方 Model 类型和实验名称外使用相同协议。SwinIR 使用 `SwinIRModel`；HINT 使用 `SRModel`。学习率的更新和恢复沿用 BasicSR 原生语义。

## 启动训练

一次选择一个实验运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync torchrun \
--nproc_per_node=2 \
--master_port=29501 \
basicsr/train.py \
-opt options/SAR/train_swinir_single.yml \
--launcher pytorch
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync torchrun \
--nproc_per_node=2 \
--master_port=29501 \
basicsr/train.py \
-opt options/SAR/train_hint_single.yml \
--launcher pytorch
```

SwinIR 与 HINT 均正式采用官方 BasicSR 的 DistributedDataParallel（DDP）。以上命令通过 `torchrun` 启动双卡 `world_size=2`；每个 rank 的 batch=3，global batch=3×2=6；每个 rank 的 `num_worker_per_gpu=4`，两个 rank 合计 workers=8。换用其他两张 GPU 时只需修改 `CUDA_VISIBLE_DEVICES`，并保持 `--nproc_per_node=2`。

后台运行示例：

```bash
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync torchrun \
--nproc_per_node=2 \
--master_port=29501 \
basicsr/train.py \
-opt options/SAR/train_swinir_single.yml \
--launcher pytorch \
> logs/swinir.console.log 2>&1 &
tail -f logs/swinir.console.log
```

训练结果分别写入 `experiments/swinir_sar_single/`、`experiments/hint_sar_single/`：`models/` 保存网络，`training_states/` 保存优化器和 scheduler，日志位于实验目录，TensorBoard 位于 `tb_logger/<实验名>/`。验证只汇总恢复 PSNR/SSIM，不逐图打印、不保存恢复图片。每次验证会暂停训练并处理完整测试集。

SwinIR 的训练内验证由各个 DDP rank 按样本索引交错分片，双卡各处理 1845 张，每卡 batch=1，不补齐或重复样本。验证前向直接使用 DDP 内部网络；结束时归约指标总和及实际样本数，按全部帧等权计算 PSNR/SSIM，然后各卡继续训练。rank 0 记录验证开始、结束、总耗时和汇总指标，NCCL 超时设置不变。

HINT 保留原有 `SRModel` 验证行为：仅 rank 0 执行全量验证，其他 rank 会进入下一训练步等待同步，长验证仍可能触发 NCCL 超时。本次 SwinIR 修复不改变 HINT 的验证顺序和持久缓存。

## 停止和续训

前台 Ctrl+C 停止；后台先用 `pgrep -af 'basicsr/train.py'` 确认对应 PID，再向该进程发送 `kill <PID>`。停止时不额外保存，以最近完整检查点为准。

续训必须显式启用官方自动恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
uv run --no-sync torchrun \
--nproc_per_node=2 \
--master_port=29501 \
basicsr/train.py \
-opt options/SAR/train_swinir_single.yml \
--launcher pytorch \
--auto_resume
```

也可在 YAML 的 `path.resume_state` 指定 `.state` 文件。网络和 training state 应成套保留。BasicSR 会恢复参数、优化器和调度状态，不保存数据迭代器；HINT 的 `qv_cache` 也不属于 BasicSR checkpoint。开启新实验时复制 YAML 并修改 `name`；不要把不带恢复选项的原命令当作续训命令。

## 独立 SAR 评估

训练内验证使用 BasicSR 的 PSNR/SSIM 指标口径，SwinIR 在各 rank 间汇总；六项输入／恢复／提升量及九帧、sequence 统计通过独立脚本生成。例如评估第 6667 步的 SwinIR：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_swinir_single.yml \
  --force_yml path:pretrain_network_g=experiments/swinir_sar_single/models/net_g_6667.pth
```

HINT 对应命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_hint_single.yml \
  --force_yml path:pretrain_network_g=experiments/hint_sar_single/models/net_g_6667.pth
```

测试配置默认指向训练完成后保存的 `net_g_latest.pth`；评估中间检查点使用上面的参数覆盖。独立评估固定单 GPU（`num_gpu: 1`）、batch 1，不使用 DDP，按 split 顺序推理，不保存恢复图。结果目录示例：

```text
results/swinir_sar_single/net_g_6667/
├── eval.log
└── sar_test/
    ├── metrics_per_frame.csv
    ├── frame_position_summary.csv
    └── metrics_per_sequence.csv
```

- 逐帧 CSV：按 `(sequence_name, frame_idx)` 排序，保留 `input/output/delta × psnr/ssim` 六项指标。
- 帧位置 CSV：固定 F₀～F₈，统计 count、六项均值与总体标准差（ddof=0）。
- sequence CSV：按名称排序，每个 sequence 包含 count 和六项均值。
- 完成后只打印一次总体指标，按全部帧等权平均。指标使用官方 uint8 image-based PSNR/SSIM、border=0，不计算 Y 通道。

绘图：

```bash
uv run --no-sync python scripts/plot_sar_metrics.py \
  results/swinir_sar_single/net_g_6667/sar_test
```

输出 `frame_position_psnr.png` 和 `frame_position_ssim.png`。这是主动生成的汇总图，与逐帧恢复图片保存无关。

## HINT 与后续扩展

HINT 保留作者源码的 persistent `qv_cache` 行为。DDP 下每个 rank 持有自己长期存在的 HINT 模型实例，因此各 rank 的 `qv_cache` 独立持续更新。本阶段不修改 cache 生命周期、不主动 reset `qv_cache`。`qv_cache` 不是 Parameter/Buffer，不属于 BasicSR checkpoint，因此 resume 后 cache 会从空状态重新开始。独立评估对每个数据集新建模型，从空 cache 开始。本节只做事实说明，不评价或修改算法。

新网络放在 `basicsr/archs/*_arch.py` 并注册 `ARCH_REGISTRY`；新数据集使用 `DATASET_REGISTRY`。仅当训练行为确有变化时才增加 `MODEL_REGISTRY` 类型。第一阶段不迁 HINT 的 FFTLoss、MixUp、渐进训练、特殊裁剪或 cyclic scheduler。

## 验证命令

```bash
uv run --no-sync python -m pytest -o addopts='' \
  tests/test_sar_dataset.py tests/test_sar_evaluation.py tests/test_hint_equivalence.py -q
```

SwinIR 的 padding、EMA 选择和验证前后模式恢复可在 CPU 上检查：

```bash
uv run --no-sync python -m pytest -o addopts='' tests/test_swinir_model.py -q
```

HINT 等价测试默认读取同级 `/data/cjz/HINT/basicsr/models/archs/HINT_arch.py`；也可通过 `HINT_REFERENCE_PATH` 指定。该文件只用于迁移测试，不是运行依赖；没有参考文件时该项跳过。

完整模型双卡短流程（确认 GPU 0、1 空闲后运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1 SAR_GPU_SMOKE=1 uv run --no-sync python -m pytest \
  -o addopts='' tests/test_sar_training.py -q -s
```

该检查使用真实数据、总 batch 6 和完整网络。SwinIR 测试实际通过 `torchrun` 启动两个 DDP 进程，检查奇数样本、空分片、重复验证、单卡指标对照、验证后继续训练和原生保存恢复；HINT 保留原有 DataParallel 短流程，不代表 HINT 的 DDP 验证已通过。只运行 SwinIR 时在上述命令末尾加 `-k swinir`。

使用已有第 6667 步检查点验证完整故障场景（需较长时间，先确认 GPU 0、1 空闲）：

```bash
check_dir=$(mktemp -d /tmp/swinir-ddp-validation.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  tests/swinir_ddp_worker.py --output "$check_dir" \
  --full-checkpoint experiments/swinir_sar_single
```

此命令为每个进程分配 16 个 CPU 线程以加快整图注意力掩码计算，读取模型、优化器和调度状态，完成 3690 张图的双卡验证后训练第 6668 步并检查两卡参数一致，然后退出。日志和 `result.json` 写入临时目录，不覆盖正式实验，也不启动长期训练。通过后可使用上文带 `--auto_resume` 的命令从正式检查点续训。

2026-09-12 验收：双 RTX 4090、每进程 16 个 CPU 线程，验证 3690 张图耗时 2709.06 秒（45 分 9 秒），PSNR=22.370765、SSIM=0.741152；随后第 6668 步 loss=0.062092，两卡全部参数完全一致。默认 `torchrun` 线程设置下的 SwinIR DDP 短流程也通过，覆盖保存恢复、连续验证、奇数/空分片及单卡指标对照。

CPU 前向、SAR 数据集和评估检查合计 9 项通过、1 项已有失败：`test_committed_split_and_shared_protocol` 要求两模型训练配置相同，但 HINT 当前配置已调整 batch、学习率和总步数。本次保留 HINT 配置及该测试，不将此失败归入 SwinIR 验证修复。
