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
| 划分 | seed 42，保留原训练集；原测试池的 20% 作验证、80% 作最终测试，九帧同属一个集合 |
| 训练 | 3,688 条 sequence，33,192 帧 |
| 验证 | 82 条 sequence，738 帧，固定 `val.txt` |
| 最终测试 | 328 条 sequence，2952 帧，独立 `test.txt`，仅在训练结束后使用 |
| 图像 | 单通道、[0,1]；训练共同裁剪 128×128，不旋转翻转；验证整图 512×512 |
| 多卡 | 官方 BasicSR DistributedDataParallel（DDP）；通过 `torchrun` + `--launcher pytorch` 启动；world_size=2；物理 GPU 0、1 |
| Batch／workers | SwinIR 每卡 batch=3、global batch=6；HINT 每卡 batch=6、global batch=12；每卡 workers=4 |
| 种子／数据重复倍率 | 42／1 |
| 损失／优化器 | Charbonnier eps=1e-9，mean；Adam，SwinIR lr=1.732e-4、HINT lr=3.464e-4；betas=[0.9,0.999]，weight decay=0 |
| 调度 | MultiStepLR，gamma=0.5；SwinIR milestones=[1066667,1600000,1866667,2000000,2133333]；HINT milestones=[533334,800000,933334,1000000,1066667] |
| 总步数 | SwinIR 2,200,000；HINT 1,100,000 |
| 日志／保存／验证 | SwinIR 每 267／6667／6667 步；HINT 每 134／3334／3334 步 |
| EMA／warmup | 关闭 |

两份配置共用数据划分与指标口径，保留各自现有训练预算。SwinIR 使用 `SwinIRModel`；HINT 使用 `SRModel`。学习率的更新和恢复沿用 BasicSR 原生语义。

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

SwinIR 与 HINT 均正式采用官方 BasicSR 的 DistributedDataParallel（DDP）。以上命令通过 `torchrun` 启动双卡 `world_size=2`；SwinIR 每个 rank 的 batch=3，global batch=6；HINT 每个 rank 的 batch=6，global batch=12；每个 rank 的 `num_worker_per_gpu=4`，两个 rank 合计 workers=8。换用其他两张 GPU 时只需修改 `CUDA_VISIBLE_DEVICES`，并保持 `--nproc_per_node=2`。

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

训练结果分别写入 `experiments/swinir_sar_single/`、`experiments/hint_sar_single/`：`models/` 保存网络，`training_states/` 保存优化器和 scheduler，日志位于实验目录，TensorBoard 位于 `tb_logger/<实验名>/`。验证只汇总恢复 PSNR/SSIM，不逐图打印、不保存恢复图片。每次验证会暂停训练，仅处理固定的 738 张验证图。

SwinIR 的训练内验证由各个 DDP rank 按样本索引交错分片，双卡各处理 369 张，每卡 batch=1，不补齐或重复样本。验证前向直接使用 DDP 内部网络；结束时归约指标总和及实际样本数，按全部帧等权计算 PSNR/SSIM，然后各卡继续训练。rank 0 记录验证开始、结束、总耗时和汇总指标，NCCL 超时设置不变。

SwinIR 的非预设尺寸掩码直接在 GPU 生成，同尺寸各层、各张图共享两张运行时掩码（512×512 时约 128 MiB）。缓存不进入检查点或 DDP buffer 同步，尺寸或设备变化时重建；128×128 训练尺寸仍使用原有 buffer。默认 CPU 单线程即可受益，无需提高线程数。

HINT 保留原有 `SRModel` 验证行为：仅 rank 0 执行 738 张图的验证，其他 rank 会进入下一训练步等待同步，长验证仍可能触发 NCCL 超时。HINT 使用新验证列表，但保持 rank 0 顺序验证与原持久缓存机制。

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

## 训练完成后的最终测试

两份训练 YAML 的 `final_test_opt` 指向各自测试配置。达到训练预算、保存 `net_g_latest.pth` 并完成末次验证后，各 rank 结束 DDP 通信并释放训练模型；仅 rank 0 启动独立单 GPU 测试进程。测试加载实际实验的最终权重，并沿用实验名称。中途保存、暂停或异常退出不触发测试。

最终测试只读取剩余 328 条 sequence（2952 帧），batch=1，按 split 顺序推理。HINT 在新进程中从空缓存开始。只保存恢复 PNG，不保存 LQ、GT 或拼接图；PNG 与指标使用相同的单通道 uint8 恢复结果。结果目录为：

```text
results/swinir_sar_single/net_g_latest/
├── eval.log
└── sar_test/
    ├── restored/<sequence>/000.png ... 008.png
    ├── metrics_per_frame.csv
    ├── frame_position_summary.csv
    └── metrics_per_sequence.csv
```

逐帧 CSV 保留 input/output/delta × PSNR/SSIM 六项指标，按 `(sequence_name, frame_idx)` 排序；帧位置与 sequence CSV 汇总均值，帧位置统计包含总体标准差（ddof=0）。总体指标按全部帧等权平均，采用 uint8 image-based PSNR/SSIM、border=0、不计算 Y 通道。

自动测试失败会报错，已经保存的最终权重保留。训练完成后可单独重跑测试：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_swinir_single.yml

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_hint_single.yml
```

自定义实验名时，通过 `--force_yml name=<实验名> path:pretrain_network_g=<实际最终权重路径>` 指定输出名称与权重。不要用最终测试集选择训练中间的检查点；训练中评估只使用验证集。

绘制最终测试指标：

```bash
uv run --no-sync python scripts/plot_sar_metrics.py \
  results/swinir_sar_single/net_g_latest/sar_test
```

## HINT 与后续扩展

HINT 保留作者源码的 persistent `qv_cache` 行为。DDP 下每个 rank 持有自己长期存在的 HINT 模型实例，因此各 rank 的 `qv_cache` 独立持续更新。本阶段不修改 cache 生命周期、不主动 reset `qv_cache`。`qv_cache` 不是 Parameter/Buffer，不属于 BasicSR checkpoint，因此 resume 后 cache 会从空状态重新开始。独立评估对每个数据集新建模型，从空 cache 开始。本节只做事实说明，不评价或修改算法。

新网络放在 `basicsr/archs/*_arch.py` 并注册 `ARCH_REGISTRY`；新数据集使用 `DATASET_REGISTRY`。仅当训练行为确有变化时才增加 `MODEL_REGISTRY` 类型。第一阶段不迁 HINT 的 FFTLoss、MixUp、渐进训练、特殊裁剪或 cyclic scheduler。

## 验证命令

```bash
uv run --no-sync python -m pytest -o addopts='' \
  tests/test_sar_dataset.py tests/test_sar_evaluation.py tests/test_sar_final_test.py -q
```

SwinIR 的 padding、EMA 选择、模式恢复和掩码复用可在 CPU 上检查：

```bash
uv run --no-sync python -m pytest -o addopts='' tests/test_swinir_model.py -q
```

HINT 等价测试默认读取同级 `/data/cjz/HINT/basicsr/models/archs/HINT_arch.py`；也可通过 `HINT_REFERENCE_PATH` 指定。该文件只用于迁移测试，不是运行依赖；没有参考文件时该项跳过。

完整模型双卡短流程（确认 GPU 0、1 空闲后运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1 SAR_GPU_SMOKE=1 uv run --no-sync python -m pytest \
  -o addopts='' tests/test_sar_training.py tests/test_sar_final_test.py -k "swinir or ddp_completion" -q -s
```

该检查使用真实数据、总 batch 6 和完整网络。SwinIR 测试实际通过 `torchrun` 启动两个 DDP 进程，检查奇数样本、空分片、重复验证、单卡指标对照、验证后继续训练和原生保存恢复；HINT 保留原有 DataParallel 短流程，不代表 HINT 的 DDP 验证已通过。上述命令只运行 SwinIR DDP 短流程与临时数据的训练完成后自动测试；最终测试集不参与验收。

使用已有完整检查点验证完整验证集（以下以第 6667 步为例，先确认 GPU 0、1 空闲）：

```bash
check_dir=$(mktemp -d /tmp/swinir-ddp-validation.XXXXXX)
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 uv run --no-sync torchrun --standalone --nproc_per_node=2 \
  tests/swinir_ddp_worker.py --output "$check_dir" \
  --resume-state experiments/swinir_sar_single/training_states/6667.state
```

此命令使用默认单线程设置，读取指定检查点的模型、优化器和调度状态，完成 738 张验证图的双卡验证后继续训练一步并检查两卡参数一致，然后退出。日志和 `result.json`（包含吞吐、耗时及显存）写入临时目录，不读取最终测试集、不覆盖正式实验，也不启动长期训练。旧检查点严格加载与手动续训保持兼容。

2026-09-12 验收使用临时短训的完整 SwinIR 检查点，评估性能和输出一致性：默认 `OMP_NUM_THREADS=1` 下，单图平均耗时从 4.62 秒降为 0.85 秒（约 5.4 倍加速）；128×128、512×512、需 padding 的 509×501 输入均与修改前逐元素一致，CPU/GPU 迁移及检查点格式检查通过。动态掩码共占 128 MiB。

双 RTX 4090 完整验证 738 张图耗时 315.91 秒（5 分 16 秒），吞吐 2.34 张/秒，每卡峰值已分配显存约 2.89 GiB；随后成功继续训练，两卡参数完全一致。15 项 CPU 检查、SwinIR DDP 短流程及真实训练入口的自动测试流程通过。最终测试集未参与本次验收。
