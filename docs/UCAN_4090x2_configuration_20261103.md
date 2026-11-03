# UCAN 双 RTX 4090 正式训练说明

## 当前配置

| 项目 | 设置 |
| --- | --- |
| 网络 | UCAN，单通道 ×1 恢复，embed_dim=48，window_size=16，mlp_ratio=1，conv_depth=5 |
| GPU／精度 | 双卡 DDP，FP32，未启用 AMP |
| Batch／crop | 每卡 6，global batch=12；128×128 配对裁剪，不旋转翻转 |
| 优化器 | Adam，lr=3.75e-4，betas=[0.9,0.99]，weight_decay=0 |
| 损失 | CharbonnierLoss，eps=1e-9，loss_weight=1，reduction=mean |
| 总预算 | 400000 iter，480 万个 patch，约 145 个 epoch |
| 调度 | MultiStepLR，milestones=[150000,250000,325000,350000,375000]，gamma=0.5 |
| 日志／保存／验证 | 每 134／10000／10000 步；验证固定 738 张 512×512 图像 |
| EMA／warmup／checkpoint | ema_decay=0，warmup_iter=-1，use_checkpoint=false |

论文参考配置由用户提供：global batch=16、64×64 crop、Adam betas=[0.9,0.99]、lr=5e-4、800k iter。当前 SAR 配置保留 128×128 crop 和共同的重建损失/增强协议，学习率按 `5e-4 × 12/16` 设置，衰减节点按论文训练进度映射到 40 万步。总 patch 曝光量与其他 SAR 基线一致；40 万步来自 batch 调整后的曝光量计算，继续从随机初始化训练。

## 数据和运行

- 先运行 QH=2.5、QL=1.5：[train_ucan_single.yml](../options/SAR/train_ucan_single.yml)，实验名 `ucan_sar_single`。
- 全 Q=2 的后续训练使用 [train_ucan_single_q2.yml](../options/SAR/train_ucan_single_q2.yml)，实验名 `ucan_sar_single_q2`。
- 对应独立测试使用 `test_ucan_single.yml` 和 `test_ucan_single_q2.yml`。

从项目根目录启动，选择两张空闲卡，以下以物理 GPU 1、3 为例：

```bash
CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync torchrun --standalone --nproc_per_node=2 basicsr/train.py \
  -opt options/SAR/train_ucan_single.yml --launcher pytorch
```

训练产物进入 `experiments/ucan_sar_single/`，TensorBoard 进入 `tb_logger/ucan_sar_single/`。每 10000 步保存权重与训练状态并运行验证；正常完成后自动测试对应数据集。中途停止可通过原配置加 `--auto_resume` 恢复最近完整检查点。

## 正式训练记录

本次从头运行 QH=2.5、QL=1.5 数据集，不加载预训练权重或旧状态。UCAN 的短测模型、状态、日志、JSON 和报告已清理；正式运行记录和日志保存在独立的正式实验目录。

2026-11-03 的启动信息见 [正式训练记录](UCAN_training_run_20261103.md)。
