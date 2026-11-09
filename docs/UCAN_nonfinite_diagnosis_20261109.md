# UCAN 首个非有限张量定位

## 排查对象

全 Q=2 实验 `ucan_sar_single_q2` 在最后正常记录的 128,640 步之后中断。
原始控制台栈在 `rg_2` 的 HybridBlock 第一处有限性检查报错；该检查位于窗口注意力和第一个 SGFN 之后，只能确定异常在此前产生或传入，不能据此认定线性注意力是根因。

120,000 步的模型参数及 Adam 状态均有限。现有训练状态未保存随机数生成器状态；不过模型的 dropout/drop path 为零，数据 worker 每个 epoch 用固定 seed 初始化 Python/NumPy，因此可以恢复采样和裁剪顺序。CUDA 非确定性计算仍可能导致续跑轨迹偏离，定位结果应以实际捕获现场为准。

## 诊断方法

独立脚本 `scripts/diagnose_ucan_nonfinite.py` 从 120k 模型和 Adam 状态续跑，保持双卡、每卡 6 个 128×128 patch、损失和学习率调度。

- 根据每个 epoch 的 2,766 个 batch，从 epoch 43 开始读取并跳过已完成的 1,062 个 batch，恢复各 worker 的裁剪序列。
- 检查前向及 loss、DDP 归约前的各参数梯度、归约后的梯度和更新后参数。
- 任一 rank 发现异常后，双方保存当前模型、优化器、实际输入/GT、样本路径和梯度异常名称，并停止续跑。
- 对失败 rank 的 batch 使用 `TorchDispatchMode` 逐算子重放，报告首个非有限算子、模块路径、输入输出的形状/非有限数量/有限元素最大幅值，以及调用栈。
- 重放时跳过 `empty` 等未初始化张量；反向使用 autograd 节点 hook 记录算子所属的反向节点。

所有现场写入 `docs/diagnostics/ucan_nonfinite/`，正式实验目录不写入新模型。

## 运行

```bash
CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  .venv/bin/torchrun --nproc_per_node=2 --master_port=29519 \
  scripts/diagnose_ucan_nonfinite.py --steps 10000 \
  > docs/diagnostics/ucan_nonfinite/replay.console.log 2>&1
```

最大运行到 130,000 步，遇到异常即提前停止。日志每 50 步记录各 rank 的 loss 和最小 Fourier 系数分母绝对值。故障文件按 `iter_<步数>_rank_<rank>` 命名，`.json` 是故障阶段与样本信息，`.pth` 是可重放现场，`_first_operator.json` 是算子定位结果。

单独重放现场时，只使用一张 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
  scripts/diagnose_ucan_nonfinite.py \
  --replay-case docs/diagnostics/ucan_nonfinite/iter_<步数>_rank_<rank>.pth
```

## 验证与当前状态

诊断器已通过 4 个 CPU 测试：有限运算与未初始化张量、前向非有限算子、有限前向对应的反向溢出、反向节点定位及 hook 清理。预期的反向溢出测试会触发 anomaly mode 的告警。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python \
  -m pytest -o addopts='' tests/test_ucan_nonfinite_diagnostic.py -q
```

## 实际捕获结果

诊断在 **129,507 步、epoch 46、rank1 前向阶段**发现异常并提前停止，共续跑 9,507 步，没有跑到 130k。rank0 同步保存现场，但该 rank 本次前向没有报错。此前各步通过了梯度及更新后参数有限性检查。

逐算子重放找到的首个非有限输出来自：

```text
rg_2.residual_group.norm_out
aten.native_layer_norm.default
```

该算子的输入形状为 `[6, 16384, 48]`，输入全部有限，最大绝对值为 `2.8575038809284438e20`；归一化权重和偏置也全部有限。算子输出包含 **336 个 NaN**，其返回的倒标准差包含 **7 个 NaN**。这是异常首次产生的位置，而非后续 HybridBlock 抛出异常的位置。

原始逐算子证据：[iter_129507_rank_1_first_operator.json](diagnostics/ucan_nonfinite/iter_129507_rank_1_first_operator.json)。双方的现场 `.pth` 保留在同一诊断目录，不纳入 Git。

### 向前追踪幅值放大路径

使用同一保存 batch 和模型参数进行 CPU 完整 batch 前向重放，异常再次出现在 `rg_2.residual_group.norm_out`。该 batch 输入范围为 `[0, 1]`，模型所有参数有限。

四个 HybridBlock 之后的特征最大绝对值约为 `2.32e2`；后续连续五个 LKSA 块的输出幅值如下。CPU 与原 GPU 重放有浮点差异，但两者均复现特征爆炸与同一 LayerNorm 失稳。

| 模块 | 输出最大绝对值 |
|---|---:|
| `blocks_2.0` | `1.87e4` |
| `blocks_2.1` | `1.52e6` |
| `blocks_2.2` | `1.92e8` |
| `blocks_2.3` | `4.98e12` |
| `blocks_2.4` | `5.17e20` |
| 最后一个 `block_merge` | `2.84e20` |

`BasicLayer.forward` 在这五轮 `SGFN → LKSA → block_merge → 残差相加` 之间没有做特征归一化。LKSA 中的核心表达式为：

```python
out = self.channel(x) * (large_kn + self.conv3(x))
```

这里的两条可学习分支直接相乘，门控值没有有界激活约束；重复堆叠会进一步放大大幅值输入。幅值增长主要发生在这一卷积分支序列中。

在同一 CPU 重放中，向 `norm_out` 输入相同的特征：

- FP32 LayerNorm 输出 240 个 NaN，均位于 batch 的第二个样本。
- 仅把这一 LayerNorm 的输入、权重和偏置改为 FP64 进行诊断计算，输出全部有限，最大绝对值约 `7.87`。
- 第二个样本为 `sar_figure_rstart_002101_cstart_000006145/006.png`。其特征最大绝对值约 `2.84e20`；其他五个样本的最大绝对值约在 `1.38e2` 到 `3.48e3` 之间。

特征本身可被 FP32 表示，但这个量级的平方达到 `1e40`，超过 FP32 有限范围。上述对照支持 LayerNorm 的统计计算在巨大有限输入下失稳，而非输入已包含 NaN 或模型权重已被 NaN 污染。

详细幅值及精度对照：[iter_129507_rank_1_amplitude_trace.json](diagnostics/ucan_nonfinite/iter_129507_rank_1_amplitude_trace.json)。

### 本次结论

本次捕获的链路是 **卷积分支连续乘法放大 → 巨大的有限特征 → FP32 LayerNorm 首次产生 NaN**。故障可以用保存的 batch 和参数再次复现，因此具有明确的数值触发条件。

现场 Fourier 系数归一化分母的最小绝对值约为 `0.04458`，此次没有捕获到 Fourier 除法直接产生非有限值。此前提出的 Fourier 分母奇异性仍是独立风险；本次直接证据把主要幅值爆炸位置定位到了五个 LKSA 块，后续修复应优先检查这段分支的归一化和乘法增益控制。

此次只补充诊断结论，没有修改网络结构或训练算法。
