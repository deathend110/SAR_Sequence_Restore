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

双卡诊断续跑已启动；当前结果以 `replay.console.log` 和现场 JSON 为准。在捕获首个异常算子之前，Fourier 带符号和归一化仍是待验证假设。
