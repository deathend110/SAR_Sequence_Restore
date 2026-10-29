# UCAN 单帧 SAR 接入记录

日期：2026-10-29。

## 接入结论

UCAN 已作为独立的单帧恢复基线接入现有 BasicSR 注册、训练和评估流程。使用官方已有的同分辨率残差恢复分支，输入和输出均为 `[B, 1, H, W]`，不经过超分上采样。该基线逐帧处理全部 9 帧，沿用本仓库 SAR 数据划分和指标协议。

来源：[官方仓库](https://github.com/hokiyoshi/UCAN)，固定版本 `b812c0adb8ce70d442973db5c87bab7d1e8c3333`。
网络文件源自 `basicsr/archs/ucan_arch.py`，Apache-2.0 许可保存在 [UCAN_LICENSE](third_party/UCAN_LICENSE)。

## 与本仓库的差异和处理

| 项目 | 官方实现 | 本次适配 |
|---|---|---|
| BasicSR 接口 | `ARCH_REGISTRY` 注册 UCAN；`UCANModel` 继承 SRModel | 保留网络注册，新建本仓库 UCANModel，复用现有 SRModel 训练流程 |
| 输入输出 | 官方测试配置为 RGB ×2/×3/×4 超分 | `in_chans: 1`、`upscale: 1`、`upsampler: null`，启用已有的残差恢复分支 |
| 网络规模 | 官方测试指定 embed_dim=48、window_size=16、mlp_ratio=1 | 保留这三个设置和 4 个残差组，SAR 版本实测 683,817 个可训练参数 |
| 位置编码 | 初始化时根据 CUDA 是否可用选择设备；频率张量不随 `.to()` 自动迁移 | CPU 初始化、注册非持久 buffer；跟随模型设备且不增加检查点持久键 |
| 辅助依赖 | 从 `timm.layers` 引入 DropPath、to_2tuple、trunc_normal_ | 复用仓库现有实现；无需增加 timm 运行依赖 |
| 窗口注意力 | 大窗口使用 PyTorch SDPA；代码允许 math 和 flash 后端 | 保留上游计算及 math 回退；显式指定 meshgrid 索引，评估时关闭注意力 dropout |
| 验证前向 | 官方 UCANModel 固定以约 200 像素切块处理；测试 YAML 的 use_chop=false 未被读取 | 整帧前向，由网络内部 padding；绕过 DDP 包装，复用现有分片和指标归约 |
| 推理状态 | 官方非 EMA 测试后直接切回 train | 保存和恢复原模式，异常时也恢复；支持 EMA 网络选择 |
| 训练配置 | 检查到的版本只提供 Test 配置，README 引用的 Train 配置未包含在仓库 | 新建两个数据集各自的 train/test YAML，使用本仓库当前 SwinIR 训练协议 |
| 评估协议 | 官方 SR 测试使用 RGB 的 Y 通道，按放大倍数裁边 | 单通道整图、crop_border=0、test_y_channel=false，沿用 uint8 image-based PSNR/SSIM |

还统一了 RotaryEmbedding 基类与子类调用的方法名，并显式导入 checkpoint 模块。保留了官方注意力缩放方式，包括 HPA 中 q 预缩放后再交给 SDPA 默认缩放的行为，以便核对上游数值。位置编码频率、相对位置索引均不进入持久权重。

当前项目环境为 Python 3.12、PyTorch 2.6.0+cu124；UCAN 使用的 `torch.nn.attention`/SDPA 接口已在该环境检查。添加 UCAN 不改变现有 SwinIR/HINT 配置或 SRModel 训练逻辑。

## 配置与公平比较

| 数据 | 训练配置 | 测试配置 | 实验名 |
|---|---|---|---|
| 原 H/L 数据集 | `options/SAR/train_ucan_single.yml` | `options/SAR/test_ucan_single.yml` | `ucan_sar_single` |
| 全 Q=2 数据集 | `options/SAR/train_ucan_single_q2.yml` | `options/SAR/test_ucan_single_q2.yml` | `ucan_sar_single_q2` |

训练配置对齐当前 SwinIR：seed42 序列划分、128×128 crop、每卡 batch=3、双卡 global batch=6、800,000 iter、Adam lr=1.732e-4、betas=[0.9,0.999]、CharbonnierLoss、EMA=0，学习率在 400k/600k/700k/760k 衰减，验证和保存间隔 20k。800k 是完整比较的预算模板，本次没有启动该训练。

保留 SARSingleDataset 的数据读取及增强约定，不加入自然图像 SR 的旋转、翻转或附加损失。训练样本曝光量保持一致后，再用短程 GPU 实测决定是否提高 UCAN batch 并同步调整学习率和 iter。当前配置采用随机初始化，不加载官方 RGB 超分权重。

## 使用方式

在项目根目录执行，下列命令为后续正式运行准备，本次未执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync torchrun \
  --nproc_per_node=2 --master_port=29503 basicsr/train.py \
  -opt options/SAR/train_ucan_single_q2.yml --launcher pytorch
```

原 H/L 数据集将配置改为 `options/SAR/train_ucan_single.yml`。产物分别进入 `experiments/ucan_sar_single_q2/` 或 `experiments/ucan_sar_single/`，训练正常结束后的自动测试沿用 `final_test_opt`。

独立评估复用现有脚本，默认读取对应实验的 `models/net_g_latest.pth`：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/eval_sar.py \
  -opt options/SAR/test_ucan_single_q2.yml
```

需要评估验证集最优检查点时，通过 `--force_yml path:pretrain_network_g=experiments/ucan_sar_single_q2/models/net_g_<iter>.pth` 覆盖权重路径，或编辑测试配置；检查点选择依据保持为验证集结果。现有脚本输出逐帧、帧位置和逐序列三份 CSV，可与 SwinIR/HINT 直接对照。

## 验证记录

CPU 回归命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python -m pytest -o addopts='' tests/test_ucan.py -q
```

检查范围：128×128、512×512 和非窗口整倍数 37×53 前向；全部可训练参数的有限梯度；checkpoint 重计算的输出和梯度一致性；EMA 选择和推理模式恢复；原生权重、优化器及调度器保存恢复后的下一次参数更新一致；CPU Gloo 双进程训练、3 张验证图的不等长分片、1 张图的空分片、分布式与独立验证指标对照，以及验证后继续训练和进程间参数一致。

最终联合回归 **24 项通过**，耗时 103.25 秒，其中 UCAN 新增 12 项；同时检查 SwinIR、HINT 和 SAR 评估脚本。两条 warning 来自现有 SwinIR 的 meshgrid/checkpoint 调用。复现联合检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python -m pytest -o addopts='' \
  tests/test_ucan.py tests/test_swinir_model.py tests/test_hint_model.py \
  tests/test_sar_evaluation.py -q
```

上游数值对照：固定 seed=42，将官方同结构权重严格加载到适配版后，单通道 ×1 恢复和 RGB ×2 超分在 37×53 输入上的输出最大绝对差均为 **0**。RGB ×2 保留官方 689,444 个参数，SAR ×1 为 683,817 个参数。

两个数据集的训练/验证路径及首条序列的 LQ/GT 图像读取均已检查，图像尺寸为 512×512；训练 3,688 条序列，验证 82 条序列，沿用同一 seed42 划分。

当前 4 张 RTX 4090 均有正在运行的任务，本次验证使用 CPU，不启动正式训练。GPU 吞吐、显存峰值及恢复精度将在后续短程实验测量，参数量不直接用作速度估计。官方逐块 NaN/Inf 检查也暂时保留；后续 GPU 性能分析需要关注这些检查触发的主机同步，以及大窗口注意力实际选择的 SDPA 后端。

建议后续顺序：双卡真实数据短程检查与吞吐测量 → 约 1k–5k iter 验证损失趋势和稳定性 → 根据相同样本曝光量安排完整对比。UCAN 的 SR 优势能否迁移到单比特 SAR，应以本数据上的验证集结果判断。
