# UCAN 3 号卡短测报告（2026-11-03）

本文件保留首次单卡短测时的设置与建议。此后完成的双卡检查及当前 YAML 参数见 [双 4090 配置报告](UCAN_4090x2_configuration_20261103.md)。

UCAN 在物理 GPU 3 的 RTX 4090 上完成单卡 FP32 短测：原 H/L 和全 Q=2 数据均可完成训练与 512×512 整帧前向，验证后继续训练、原生保存恢复均通过。batch=6 的单卡吞吐比 batch=3 提高约 35.4%，可作为后续双卡短测的候选。

## 范围与设置

- 只使用物理 GPU 3，UUID 为 `GPU-d0efb9f4-38ae-8c58-466f-6c4a3cc36e88`，进程内映射为 `cuda:0`。开始前该卡空闲。
- 时间：2026-11-03 16:01:02～16:02:21（Asia/Shanghai），含两次进程启动与阶段间隔约 79 秒。
- 来源提交：`6980bcd`；PyTorch 2.6.0+cu124。只用一张卡，`num_gpu=1`、`dist=false`，未执行 DDP。
- 保留当前 UCAN 网络：单通道 ×1 恢复、embed_dim=48、window_size=16、mlp_ratio=1、conv_depth=5、use_checkpoint=false，共 683,817 个参数。
- 保留 128×128 训练 crop、seed42、每卡 4 workers、Adam lr=1.732e-4、betas=[0.9,0.999]、Charbonnier eps=1e-9。随机初始化，未加载已有恢复权重。
- FP32，不启用 AMP；cuDNN benchmark=true、matmul TF32=false、cuDNN TF32=true，与当前训练入口的默认后端设置一致。
- 原 H/L 使用 `train_ucan_single.yml`，全 Q=2 使用 `train_ucan_single_q2.yml`；只在内存中覆盖单卡运行、实验输出路径及 batch。正式 YAML 和生产代码保持原样。

## 训练速度和显存

稳态计时包含取数据、传输、前向、反向、Adam 更新及 CUDA 同步。H/L 各档排除前 10 步；显存峰值统计包含启动/预热阶段。这里的 allocated/reserved 是 PyTorch 口径。

| 数据 | Batch | 训练步数 | 测速步数 | 秒/步 | Patch/秒 | 峰值 allocated GiB | 峰值 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 H/L | 3 | 50 | 40 | 0.3050 | 9.84 | 10.22 | 10.69 |
| 原 H/L | 6 | 30 | 20 | 0.4505 | 13.32 | 18.95 | 20.51 |
| 全 Q=2 | 3 | 5 | 4 | 0.3103 | 9.67 | 9.08 | 10.69 |

Q=2 仅用于数据与执行流程检查，其 4 个计时步作为辅助记录。H/L 两档独立进程测速，batch=6 每步更长，但单位时间处理的 patch 更多；取数据平均约 0.0021～0.0022 秒/步。

原 H/L、batch=3 的首步 loss 为 0.259597，第 50 步为 0.075063；batch=6 的首步为 0.269262，第 30 步为 0.108771。全部 loss 有限，首末步所有可训练参数均有有限梯度，16 个位置编码频率 buffer 均位于当前 CUDA 设备。不同 batch 的训练曝光量不同，本轮用这些 loss 检查数值稳定性。

## 整帧验证与保存恢复

只读取验证集首条序列 `port_rstart_002701_cstart_000013377`，原 H/L 检查 F0/F4/F8，全 Q=2 检查 F4。调用当前 UCANModel.test，整帧前向后输出为 `[1,1,512,512]`、值有限，原训练模式恢复正常。

| 数据 | 帧位置 | 单帧前向耗时（秒） | 验证峰值 allocated GiB | 验证峰值 reserved GiB |
| --- | --- | --- | ---: | ---: |
| 原 H/L | F0、F4、F8 | 0.8604、0.4855、0.3956 | 7.23 | 12.79 |
| 全 Q=2 | F4 | 0.4021 | 7.23 | 9.77 |

首张整帧输入包含该尺寸首次执行的开销。耗时只统计模型前向，不包含数据解码或 PSNR/SSIM 计算。

原 H/L 在 50 步训练和验证后继续第 51 步，保存 `net_g_51.pth` 与 `51.state`。重新构建模型后，权重、Adam 状态和 scheduler 状态逐项严格一致；恢复后完成第 52 步，loss=0.063293。Q=2 验证后继续第 6 步，loss=0.163486。

本轮主循环共 85 步，加验证后继续训练及恢复检查的 3 步，合计 88 次参数更新、354 个训练 patch。没有使用最终测试集，也未启动完整训练预算。

## 配置建议

当前每卡 batch=3 已通过单卡真实数据验证，可继续用作双卡起始配置。batch=6 在这张卡上通过 30 步短测，吞吐有提升，峰值 reserved 约 20.51 GiB；下一步可在两张空闲卡上测 DDP、分片验证及保存恢复，再决定是否采用每卡 6。

本轮验证流程与性能，恢复精度通过后续验证集实验评估。正式配置仍为 SwinIR/UCAN 每卡 batch=3、800000 iter，HINT 每卡 batch=6、400000 iter。如果以后 UCAN 改为双卡每卡 6，保持 480 万个 patch 曝光量时对应 400000 iter，学习率和调度节点需一并安排。

## 证据与本地复现

- [完整结果和逐步 loss/计时](experiments/ucan_gpu3_smoke_20261103/results_summary.json)：包含正式配置 SHA256、来源提交及本地脚本 SHA256。
- [首轮控制台日志](../experiments/ucan_gpu3_smoke_20261103/console.log)、[batch=6 日志](../experiments/ucan_gpu3_smoke_20261103/batch6_console.log)。
- [首轮脚本](../experiments/ucan_gpu3_smoke_20261103/run_smoke.py)、[追加 batch=6 脚本](../experiments/ucan_gpu3_smoke_20261103/run_batch6.py)；脚本和权重保存在本机独立实验目录，Git 中保留报告及 JSON 结果。

在确认 3 号卡空闲后，从仓库根目录复现本地脚本：

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=GPU-d0efb9f4-38ae-8c58-466f-6c4a3cc36e88 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u \
  experiments/ucan_gpu3_smoke_20261103/run_smoke.py

timeout 180s env CUDA_VISIBLE_DEVICES=GPU-d0efb9f4-38ae-8c58-466f-6c4a3cc36e88 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u \
  experiments/ucan_gpu3_smoke_20261103/run_batch6.py
```

首轮脚本使用保守显存门槛，可能跳过 batch=6；本次检查首轮结果后另行执行了追加脚本。复现会覆盖该独立目录内同名结果，需先保存现有产物。
