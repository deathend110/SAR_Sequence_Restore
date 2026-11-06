# UCAN 全 Q=2 对比实验正式训练

启动时间：2026-11-06 16:00:10（Asia/Shanghai）。

本轮使用全 Q=2 数据集从头训练，与已完成的 QH=2.5、QL=1.5 训练保持相同模型和训练参数。逐项比较上一轮保存的 `experiments/ucan_sar_single/train_ucan_single.yml`，差异仅为实验名、LQ/GT 数据路径和最终测试配置路径。当前损失仍为 Charbonnier，小波损失仅注册备用。

| 项目 | 设置 |
| --- | --- |
| 实验名 | `ucan_sar_single_q2` |
| 配置 | [train_ucan_single_q2.yml](../options/SAR/train_ucan_single_q2.yml) |
| 数据集 | `range_2d_sft_q2_s0_fr1p3_fa0p4_phi0_lp0p99_hp99p9_roi600` |
| GPU | 物理 GPU 1、3；两张 RTX 4090，双进程 DDP |
| 初始化 | 随机初始化，`pretrain_network_g=null`、`resume_state=null`；从 epoch=0、iter=0 开始 |
| 随机种子／划分 | seed=42；沿用 `docs/splits/sar_single_seed42/` |
| 训练／验证数据 | 33192 帧／738 帧；已检查全部配对文件存在，并读取首个样本验证尺寸 |
| Batch／crop | 每卡 6，全局 12；128×128 随机裁剪，不翻转或旋转 |
| 网络 | UCAN 同分辨率单通道恢复分支；embed_dim=48、window_size=16 |
| 训练预算 | 400000 iter，480 万个 patch；145 个训练 epoch 的循环预算 |
| 优化器 | Adam；lr=3.75e-4，betas=[0.9,0.99]，weight_decay=0 |
| 学习率调度 | MultiStepLR；150k/250k/325k/350k/375k 时乘以 0.5 |
| 损失 | CharbonnierLoss，loss_weight=1，eps=1e-9，reduction=mean |
| 其他 | FP32，EMA=0，warmup_iter=-1，use_checkpoint=false；每卡 workers=4 |
| 保存／验证／日志 | 每 10000 步保存与验证；每 134 步记录日志 |
| 后台启动器 PID | `3792969` |
| 启动代码提交 | `b914cf9db2864f4ee7be690dff531e795217a349` |
| 配置 SHA256 | `8fcafe075397d6423b4a9c2ea71516fe9cd7a11a37a49b064ac1c16ea976986e` |

启动命令如下，实际启动使用 GPU UUID 固定物理卡映射，并设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`PYTHONUNBUFFERED=1`：

```bash
CUDA_VISIBLE_DEVICES=GPU-dacf23c1-c0bd-188d-aa2b-f33bf8dc372d,GPU-d0efb9f4-38ae-8c58-466f-6c4a3cc36e88 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
.venv/bin/torchrun --standalone --nproc_per_node=2 basicsr/train.py \
  -opt options/SAR/train_ucan_single_q2.yml --launcher pytorch
```

进程使用独立后台会话，标准输入关闭，输出重定向到控制台日志，关闭当前终端不会中断训练。启动前确认两张卡没有计算进程；GPU 0、2 的其他任务保持运行。

启动检查：2026-11-06 16:01:50 已输出第 134 步，lr=`3.750e-4`、`l_pix=0.065681`，损失为有限值。两张卡显存各约 21622 MiB、利用率正常。控制台有 PyTorch DDP 梯度 stride 与 bucket view 不同的性能提示，训练已正常推进，未发现 OOM 或启动异常。

## 日志与产物

- [正式训练日志](../experiments/ucan_sar_single_q2/train_ucan_sar_single_q2_20261106_160014.log)。
- [后台控制台日志](../logs/ucan_sar_single_q2_20261106_160010.console.log)。
- [保存的配置快照](../experiments/ucan_sar_single_q2/train_ucan_single_q2.yml)。
- [启动元数据](../experiments/ucan_sar_single_q2/launch.json)：含完整命令、GPU UUID、PID、配置哈希和代码提交。
- 权重：`experiments/ucan_sar_single_q2/models/`。
- 恢复状态：`experiments/ucan_sar_single_q2/training_states/`。
- TensorBoard：`tb_logger/ucan_sar_single_q2/`。
- 正常完成后自动执行 [test_ucan_single_q2.yml](../options/SAR/test_ucan_single_q2.yml)，使用本轮最终权重评估并保存恢复图。

监控命令：

```bash
tail -f experiments/ucan_sar_single_q2/train_ucan_sar_single_q2_20261106_160014.log
```

如进程意外停止，确认没有同名训练进程且 GPU 空闲后，可沿用上面的启动命令并追加 `--auto_resume`，读取本实验自己的恢复状态。

## 时间参考

上一轮于 2026-11-06 02:38 结束训练，日志记录训练阶段耗时 58 小时 9 分；最终测试于 03:05 完成。本轮相同硬件、训练预算与数据规模，预计约 58～60 小时，参考完成时间为 2026-11-09 02:00～04:00（Asia/Shanghai），实际取决于训练速度和验证耗时。
