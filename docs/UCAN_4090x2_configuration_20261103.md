# UCAN 双 RTX 4090 训练配置（2026-11-03）

本次按论文训练参数和 SAR 数据协议选择双卡配置，短程训练、整帧分片验证与原生保存恢复均已通过，原 H/L、全 Q=2 两份 YAML 已更新。

## 配置取舍

论文参数来自用户提供的第 4.2 节：batch=16、64×64 crop、Adam betas=[0.9,0.99]，×2 从头训练 800k iter、lr=5e-4、milestones=[300k,500k,650k,700k,750k]。论文同时使用 L1、LDL、Wavelet 损失及自然图像旋转/翻转。

| 项目 | SAR 原起始配置 | 当前正式配置 | 依据 |
| --- | --- | --- | --- |
| GPU／精度 | 双卡／FP32 | 双 4090／FP32 | 本轮使用空闲物理 GPU 1、3，保持未启用 AMP |
| 单卡／global batch | 3／6 | 6／12 | 保留 128×128 crop，实际测试显存和吞吐 |
| Crop | 128×128 | 128×128 | 与现有 SAR 基线一致；每个 patch 像素数是论文 64×64 的 4 倍 |
| 初始学习率 | 1.732e-4 | 3.75e-4 | 参考论文 global batch 比例：5e-4 × 12/16；作为短测验证的工程起点 |
| Adam betas | [0.9,0.999] | [0.9,0.99] | 采用论文 beta2 |
| 总 iter | 800000 | 400000 | global batch 翻倍，总曝光量保持 480 万个 patch，约 145 个 epoch |
| Milestones | [400k,600k,700k,760k] | [150k,250k,325k,350k,375k] | 论文各衰减节点对应训练进度 37.5%/62.5%/81.25%/87.5%/93.75% |
| 调度倍率 | 0.5 | 0.5 | 沿用论文逐次减半 |
| 验证／保存间隔 | 20000／20000 | 10000／10000 | 均每 12 万个训练 patch 进行一次 |
| 日志间隔 | 267 | 134 | 约每 1600 个训练 patch 记录一次 |
| 损失 | Charbonnier | Charbonnier，eps=1e-9 | 与 SwinIR/HINT 共用重建损失；该形式接近 L1 |
| 增强 | 配对裁剪，不旋转翻转 | 保持一致 | 保留 SAR 轴向和对应关系 |
| EMA／warmup／checkpoint | 关闭 | 关闭 | 先验证当前 FP32 计算流程 |

这套方案使用 UCAN 专属的 Adam 参数及调度比例，并保持 SAR 数据划分、crop、损失和总 patch 曝光量一致。论文的 ×3/×4 迁移学习针对超分倍率，本任务为单通道 ×1 恢复，继续从随机初始化训练。40 万步来自 batch 翻倍后保持本项目曝光量的计算，与论文高倍率模型的 400k 微调策略无关。

Charbonnier 定义为 `sqrt(error² + eps)`，当前 eps=1e-9 对应平滑尺度约 3.16e-5。LDL、Wavelet 不在本轮加入，后续若研究论文完整损失组合，应作为单独消融记录。

## 短测与验证

训练计时包括数据读取、传输、前后向、优化器更新及 DDP 同步，排除前 10 步；显存峰值包含预热阶段，采用 PyTorch allocated 口径。物理 GPU 1、3 在测试开始时空闲。

| 配置 | 每卡 batch | Global batch | 测试步数 | 秒/步 | 双卡 patch/秒 | 每卡峰值 allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原起始配置 | 3 | 6 | 50 | 0.3327 | 18.03 | 10.22 |
| 论文参数适配配置 | 6 | 12 | 200 | 0.4918 | 24.40 | 18.95 |

新方案吞吐提高约 35.3%；每卡峰值 reserved 为 20.51 GiB，给 24 GiB 显卡保留一定余量，因此本轮选择每卡 6。200 步 loss 从 0.316993 降到 0.058202，首末步所有参数梯度有限、两卡参数一致。

首次 200 步作业在验证指标严格对照处停止，保留日志为 `paper_b6_initial.console.log`。复核中记录到默认后端下分布式与独立验证的 PSNR 最大差值约 6.41e-6 dB；对照时固定 cuDNN 的 benchmark=false、deterministic=true 后，保持 1e-6 的断言容差通过。训练和完整验证仍使用原默认后端。短测也出现一次 DDP 梯度 stride 警告，未影响梯度有限性与两卡参数同步检查。

完整验证集 82 条序列、738 张 512×512 图像全部处理完成，双卡各 369 张，耗时 **246.78 秒**；验证峰值 allocated 为 **7.23 GiB/卡**。第 200 步完整验证记录为 PSNR=19.92349 dB、SSIM=0.673576，用于确认早期训练和验证流程，恢复精度由后续训练的验证曲线评价。

3 张图的不等长分片、1 张图的空分片及独立参考指标对照通过。完整验证后成功继续第 201 步，loss=0.073852；保存 `net_g_201.pth` 与 `201.state`，重新构建模型后，权重、Adam 状态及调度器状态逐项一致，恢复后第 202 步 loss=0.065729，两卡参数仍严格同步。

随后直接读取更新后的 Q=2 正式 YAML，完成 20 步训练、相同的短分片验证、验证后第 21 步和恢复后第 22 步。Q=2 双卡吞吐约 24.50 patch/秒、峰值 allocated 18.95 GiB/卡；恢复后 loss=0.086148。未读取最终测试集，未启动完整训练。

9 项针对配置协议、有限梯度、checkpoint 重计算、推理模式、保存恢复和 CPU Gloo 分片验证的回归检查通过，耗时 21.32 秒：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python -m pytest -o addopts='' tests/test_ucan.py \
  -k 'not full_frame_and_non_window_size' -q
```

## 时间估计

按当前双卡稳态计时估算，40 万步纯训练约 **54.64 小时**，40 次完整周期验证约 **2.74 小时**，合计约 **57.38 小时/数据集**。完整运行还包含保存、末次验证及最终独立测试；实际安排可按约 **2～3 天/数据集** 预留。两套数据顺序训练约 4～6 天。

原每卡 3、80 万步配置的纯训练估计约 73.94 小时；本次保持相同总 patch 曝光量，纯训练时间减少约 19.3 小时。以上为本轮硬件状态下的短测外推，不把短测指标当作长期最优参数的证据。

## 正式配置与启动

- 原 H/L：[train_ucan_single.yml](../options/SAR/train_ucan_single.yml)，实验名 `ucan_sar_single`。
- 全 Q=2：[train_ucan_single_q2.yml](../options/SAR/train_ucan_single_q2.yml)，实验名 `ucan_sar_single_q2`。
- 测试网络结构和权重格式相同，继续使用对应 `test_ucan_single*.yml`。

先确认两张卡空闲，以下以物理 GPU 1、3 为例：

```bash
CUDA_VISIBLE_DEVICES=1,3 uv run --no-sync torchrun \
  --nproc_per_node=2 --master_port=29503 basicsr/train.py \
  -opt options/SAR/train_ucan_single.yml --launcher pytorch
```

Q=2 使用 `options/SAR/train_ucan_single_q2.yml`。本次只执行短测，不启动上述完整训练。新 recipe 从头训练；已有 `.state` 会恢复其优化器和调度状态，不能用旧状态直接检验新的 recipe。

## 复现与证据

- [短测步骤、命令与参数说明](experiments/ucan_ddp_tuning_20261103/README.md)。
- [双卡检查脚本](experiments/ucan_ddp_tuning_20261103/run_check.py)。
- [结果、逐步计时与配置摘要](experiments/ucan_ddp_tuning_20261103/results_summary.json)。
- 本机原始产物位于 `experiments/ucan_ddp_tuning_20261103/`，正式训练目录与历史权重独立保留。
