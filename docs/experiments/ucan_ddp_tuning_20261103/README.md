# UCAN 双 RTX 4090 配置短测（2026-11-03）

使用物理 GPU 1、3，两张卡开始时均空闲；GPU 0、2 上的其他任务保持运行。只执行有界短测，不启动完整训练。网络为完整 UCAN，训练 crop=128、FP32，不启用 AMP，验证整图 512×512。

## 方案与论文参数适配

论文参考设置由用户提供：global batch=16、crop=64、Adam betas=[0.9,0.99]、lr=5e-4、800k iter，衰减节点为 [300k,500k,650k,700k,750k]，损失包含 L1、LDL、Wavelet，并使用旋转/翻转。

SAR 方案保持 crop=128、单通道 ×1 恢复、不旋转翻转，以及各基线共用的 CharbonnierLoss。双卡每卡 batch=6 时 global batch=12，参考学习率按 batch 比例取 `5e-4 × 12/16 = 3.75e-4`，采用论文的 beta2=0.99。总预算设为 400k，保持与其他 SAR 基线一致的 480 万个 patch；按论文训练进度比例设置 milestones=[150k,250k,325k,350k,375k]，gamma=0.5。这是 SAR 适配方案，学习率缩放作为待验证的工程起点。

## 执行步骤

1. 原配置每卡 batch=3，50 步，排除前 10 步测吞吐。
2. 候选配置每卡 batch=6，200 步，排除前 10 步；检查首末步梯度、全部参数同步、3 张图的不等长分片、1 张图的空分片、独立验证指标对照、全部 738 张验证图、验证后训练及原生保存恢复。
3. 更新正式原 H/L、Q=2 配置，再从 Q=2 配置直接进行 20 步双卡检查和短分片验证、保存恢复。
4. 核对配置、CPU 回归、结果重算和文档，保存报告后退出。

分布式与独立验证先使用默认后端对照；若出现超过 1e-6 的指标差值，各 rank 统一改用 benchmark=false、deterministic=true 的 cuDNN 设置重新做严格对照，结束后恢复训练后端。这样将 GPU 算法选择带来的微小差异与分片/归约逻辑检查分开记录，完整验证耗时仍测默认后端。首轮对照停止的日志另存为 `paper_b6_initial.console.log`，补充诊断作业为 `diagnostic_b6`，最终结果使用复核后的 `paper_b6`。

## 本地复现

从项目根目录执行，先确认两张卡空闲。使用 GPU UUID 固定物理设备：

```bash
UCAN_TEST_GPUS=GPU-dacf23c1-c0bd-188d-aa2b-f33bf8dc372d,GPU-d0efb9f4-38ae-8c58-466f-6c4a3cc36e88
```

正式 YAML 更新后，原 baseline 从调参前提交导出：

```bash
git show 84ae690:options/SAR/train_ucan_single.yml > /tmp/ucan_before_tuning_20261103.yml
timeout 240s env CUDA_VISIBLE_DEVICES="$UCAN_TEST_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  docs/experiments/ucan_ddp_tuning_20261103/run_check.py \
  --config /tmp/ucan_before_tuning_20261103.yml --label baseline_b3 --batch 3 --steps 50
```

候选及 Q=2 检查：

```bash
timeout 600s env CUDA_VISIBLE_DEVICES="$UCAN_TEST_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  docs/experiments/ucan_ddp_tuning_20261103/run_check.py \
  --config options/SAR/train_ucan_single.yml --label paper_b6 --batch 6 --steps 200 \
  --paper-adapted --check-flow --full-val

timeout 240s env CUDA_VISIBLE_DEVICES="$UCAN_TEST_GPUS" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  docs/experiments/ucan_ddp_tuning_20261103/run_check.py \
  --config options/SAR/train_ucan_single_q2.yml --label q2_final_b6 --batch 6 --steps 20 --check-flow
```

每项运行保存 runtime_options.json、各 rank 日志及 result.json 到 `experiments/ucan_ddp_tuning_20261103/<label>/`。再次运行同标签会覆盖对应短测产物，复现前另存原结果。最终汇总保存到本目录，与报告一并提交。
