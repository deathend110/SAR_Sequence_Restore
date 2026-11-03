# UCAN QH=2.5、QL=1.5 正式训练

启动时间：2026-11-03 16:28:07（Asia/Shanghai）。

| 项目 | 设置 |
| --- | --- |
| 实验名 | ucan_sar_single |
| 配置 | [train_ucan_single.yml](../options/SAR/train_ucan_single.yml) |
| 数据 | range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64 |
| GPU | 物理 GPU 1、3；双进程 DDP |
| 初始化 | 从头训练，pretrain_network_g=null、resume_state=null，epoch=0、iter=0 |
| Batch／crop | 每卡 6，global batch=12；128×128 |
| 优化器／预算 | Adam，lr=3.75e-4，betas=[0.9,0.99]；400000 iter |
| 保存／验证 | 每 10000 步；固定 738 张验证图 |
| 启动器 PID | 188704 |

启动检查：2026-11-03 16:29:47 已记录第 134 步，学习率 3.75e-4，l_pix=0.069382；进程在独立后台会话运行，未出现训练异常。

已删除两个 UCAN 短测实验目录中的 7 份权重、7 份训练状态及其他运行产物，并清理对应短测报告、JSON 和临时检查脚本。网络代码、正式 YAML、回归测试及训练说明保留。

## 日志和产物

- [后台控制台日志](../logs/ucan_sar_single_20261103_162807.console.log)。
- [正式训练日志](../experiments/ucan_sar_single/train_ucan_sar_single_20261103_162812.log)。
- [启动元数据](../experiments/ucan_sar_single/launch.json)：包含命令、GPU UUID、PID、配置 SHA256 和代码来源提交。
- 权重：`experiments/ucan_sar_single/models/`。
- 恢复状态：`experiments/ucan_sar_single/training_states/`。
- TensorBoard：`tb_logger/ucan_sar_single/`。

进程采用独立会话在后台运行，关闭当前终端不会中断训练。日志监控：

```bash
tail -f logs/ucan_sar_single_20261103_162807.console.log
```

中途停止后的恢复命令：

```bash
CUDA_VISIBLE_DEVICES=1,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync torchrun --standalone --nproc_per_node=2 basicsr/train.py \
  -opt options/SAR/train_ucan_single.yml --launcher pytorch --auto_resume
```

达到训练预算后，框架保存最终权重并自动执行 `test_ucan_single.yml`。正式运行进度以当前日志为准。
