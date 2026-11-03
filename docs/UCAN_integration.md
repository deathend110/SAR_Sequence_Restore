# UCAN 单帧 SAR 接入记录

日期：2026-10-29。

参数核对日期：2026-11-03；三种模型的当前参数总表见 [README_SAR.md](README_SAR.md#数据和训练协议)，执行时以正式 YAML 为准。

## 接入结论

UCAN 已作为独立的单帧恢复基线接入现有 BasicSR 注册、训练和评估流程。使用官方已有的同分辨率残差恢复分支，输入和输出均为 `[B, 1, H, W]`，不经过超分上采样。该基线逐帧处理全部 9 帧，沿用本仓库 SAR 数据划分和指标协议。

来源：[官方仓库](https://github.com/hokiyoshi/UCAN)，固定版本 `b812c0adb8ce70d442973db5c87bab7d1e8c3333`。
网络文件源自 `basicsr/archs/ucan_arch.py`，Apache-2.0 许可保存在 [UCAN_LICENSE](third_party/UCAN_LICENSE)。

## 与本仓库的差异和处理

| 项目 | 官方实现 | 本次适配 |
|---|---|---|
| BasicSR 接口 | `ARCH_REGISTRY` 注册 UCAN；`UCANModel` 继承 SRModel | 保留网络注册，新建本仓库 UCANModel，复用现有 SRModel 训练流程 |
| 输入输出 | 官方测试配置为 RGB ×2/×3/×4 超分 | `in_chans: 1`、`upscale: 1`、`upsampler: null`，启用已有的残差恢复分支 |
| 网络规模 | 官方测试指定 embed_dim=48、window_size=16、mlp_ratio=1 | 保留这三个设置和 4 个残差组，SAR 版本共 683,817 个可训练参数 |
| 位置编码 | 初始化时根据 CUDA 是否可用选择设备；频率张量不随 `.to()` 自动迁移 | CPU 初始化、注册非持久 buffer；跟随模型设备且不增加检查点持久键 |
| 辅助依赖 | 从 `timm.layers` 引入 DropPath、to_2tuple、trunc_normal_ | 复用仓库现有实现；无需增加 timm 运行依赖 |
| 窗口注意力 | 大窗口使用 PyTorch SDPA；代码允许 math 和 flash 后端 | 保留上游计算及 math 回退；显式指定 meshgrid 索引，评估时关闭注意力 dropout |
| 验证前向 | 官方 UCANModel 固定以约 200 像素切块处理；测试 YAML 的 use_chop=false 未被读取 | 整帧前向，由网络内部 padding；绕过 DDP 包装，复用现有分片和指标归约 |
| 推理状态 | 官方非 EMA 测试后直接切回 train | 保存和恢复原模式，异常时也恢复；支持 EMA 网络选择 |
| 训练配置 | 检查到的版本只提供 Test 配置，README 引用的 Train 配置未包含在仓库 | 新建两个数据集各自的 train/test YAML；数据、损失与预算沿用 SAR 对比协议，优化器与调度参考论文 |
| 评估协议 | 官方 SR 测试使用 RGB 的 Y 通道，按放大倍数裁边 | 单通道整图、crop_border=0、test_y_channel=false，沿用 uint8 image-based PSNR/SSIM |

还统一了 RotaryEmbedding 基类与子类调用的方法名，并显式导入 checkpoint 模块。保留了官方注意力缩放方式，包括 HPA 中 q 预缩放后再交给 SDPA 默认缩放的行为，以便核对上游数值。位置编码频率、相对位置索引均不进入持久权重。

当前项目环境为 Python 3.12、PyTorch 2.6.0+cu124；UCAN 使用的 `torch.nn.attention`/SDPA 接口已在该环境检查。添加 UCAN 不改变现有 SwinIR/HINT 配置或 SRModel 训练逻辑。

## 配置与公平比较

| 数据 | 训练配置 | 测试配置 | 实验名 |
|---|---|---|---|
| 原 H/L 数据集 | `options/SAR/train_ucan_single.yml` | `options/SAR/test_ucan_single.yml` | `ucan_sar_single` |
| 全 Q=2 数据集 | `options/SAR/train_ucan_single_q2.yml` | `options/SAR/test_ucan_single_q2.yml` | `ucan_sar_single_q2` |

当前正式训练配置：seed42 序列划分、128×128 crop、每卡 batch=6、双卡 global batch=12、每卡 workers=4、400,000 iter、Adam lr=3.75e-4、betas=[0.9,0.99]、weight_decay=0；CharbonnierLoss 的 eps=1e-9、loss_weight=1、reduction=mean。FP32，未启用 AMP，EMA=0、warmup_iter=-1，学习率在 150k/250k/325k/350k/375k 乘以 0.5，日志间隔 134 步，验证和保存间隔 10k。两套数据配置的这些参数相同。400k 对应 480 万个 patch、约 145 个 epoch，与其他 SAR 基线保持总曝光量一致；论文参数的适配方式见 [双 4090 训练说明](UCAN_4090x2_configuration_20261103.md)。

保留 SARSingleDataset 的数据读取及增强约定，不加入自然图像 SR 的旋转、翻转或附加损失。UCAN 使用参考论文适配的优化器和调度参数，数据划分、crop、重建损失与总曝光量和其他 SAR 基线一致。当前配置采用随机初始化，不加载官方 RGB 超分权重。

## 使用方式

在项目根目录执行；以下为原 H/L 数据集的正式训练命令：

```bash
CUDA_VISIBLE_DEVICES=1,3 uv run --no-sync torchrun \
  --nproc_per_node=2 --master_port=29503 basicsr/train.py \
  -opt options/SAR/train_ucan_single.yml --launcher pytorch
```

全 Q=2 数据集将配置改为 `options/SAR/train_ucan_single_q2.yml`。产物分别进入 `experiments/ucan_sar_single_q2/` 或 `experiments/ucan_sar_single/`，训练正常结束后的自动测试沿用 `final_test_opt`。

独立评估复用现有脚本，默认读取对应实验的 `models/net_g_latest.pth`：

```bash
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_ucan_single.yml
```

需要评估验证集最优检查点时，通过 `--force_yml path:pretrain_network_g=experiments/ucan_sar_single_q2/models/net_g_<iter>.pth` 覆盖权重路径，或编辑测试配置；检查点选择依据保持为验证集结果。现有脚本输出逐帧、帧位置和逐序列三份 CSV，可与 SwinIR/HINT 直接对照。

## 回归检查

网络、模型适配和 CPU Gloo 分片验证的回归测试保留在 `tests/test_ucan.py`：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python -m pytest -o addopts='' tests/test_ucan.py -q
```

覆盖输入尺寸、有限梯度、checkpoint 重计算、EMA/模式恢复、原生保存恢复及不等长/空分片验证。短测运行产物与报告已按要求清理，正式训练参数见 YAML，实际训练进度见实验日志。
