# SAR 工作区迁移记录

实施日期：2026-09-12。新工作区：`/data/cjz/SAR_Sequence_Restore`。

## 来源与原则

| 来源 | 使用内容 |
| --- | --- |
| `/data/cjz/BasicSR` | 本地 BasicSR 1.4.2 框架、原生 Model／训练／指标／保存恢复、示例与许可证 |
| `/data/cjz/SAR_Transformer` | SAR Dataset、seed 42 的 9:1 split、配置参数、CSV 统计和绘图、问题定义 |
| `/data/cjz/HINT` | `basicsr/models/archs/HINT_arch.py` 和 Rain100L YAML 的核心结构参数 |

BasicSR 来源目录没有 Git 历史，本项目以实际复制的源码建立独立基线提交 `efd9e83`；它不是上游提交号。HINT 来源为作者项目 [joshyZhou/HINT](https://github.com/joshyZhou/HINT)，对应论文 *Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration*。复制时保留原源码标识；本地 HINT 来源目录未附独立 LICENSE 文件。BasicSR 原有许可证文件保留在工作区。

执行优先级：用户最终要求、标准 BasicSR 机制、迁移地图。未复制 KAIR 权重，不实现旧权重兼容，不迁 HINT fork 的 Model、训练入口或特殊 recipe。

复制来源目录仅用于读取，数据仍在原 SAR 数据目录。训练与评估只导入本工作区的 BasicSR，参考源码只在迁移测试中使用。

## 实现

- `SARSingleDataset` 注册到官方 DATASET_REGISTRY，保留 sequence 九帧展开、同名帧配对、灰度归一化、共同裁剪、不旋转翻转及 metadata。
- split 原样迁入，训练 3,688 条／33,192 帧，验证测试 410 条／3,690 帧，seed 42；没有新增独立 val 划分。
- SwinIR 使用官方 SwinIRModel，HINT 使用官方 SRModel，训练使用原生 DataParallel；没有自定义 Model 包装。
- 两份训练 YAML 使用同一协议：总 batch 6、patch 128、Adam 1.732e-4、Charbonnier eps=1e-9、原 milestones、总预算 220 万步。
- 训练内验证使用官方恢复 PSNR/SSIM 汇总，save_img=false、pbar=false。独立评估输出输入／恢复／delta 六项指标及逐帧、帧位置、sequence 三份 CSV。
- 91 个来源 BasicSR Python 文件经过逐字节比较，内容全部保持一致；新增网络与数据集独立注册。

### HINT 的实际 DP 适配

首次完整双卡运行时，原源码的 `self.inter_modulators` 普通 Python 列表在第二张卡仍引用主卡子模块，产生 `cuda:1` 输入与 `cuda:0` 卷积权重不一致的错误。

迁移版删除这份重复列表，通过 `getattr(self, f'inter_modulator{i + 1}')` 读取当前 DP 副本中已经注册的四个子模块。保留所有参数名、shape、核心计算和 qv_cache 更新公式。修正后原版与迁移版同权重、同输入的输出和 cache 等价测试通过，完整双卡训练也通过。

HINT 的普通属性 cache 不属于 BasicSR checkpoint；DP 副本生命周期沿用框架。reset／persistent、输入顺序影响与恢复时 cache 策略未引入本轮训练方案。

## 验证结果

环境：Python 3.12.14、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、NumPy 1.26.4；可编辑导入路径为新工作区 `basicsr/__init__.py`。

- 十三项常规测试通过，其中七项 SAR／HINT 测试覆盖：数据集展开与灰度、paired crop 与 DP batch、非法 split／缺失配对、图像尺寸、正式 split 与共享配置、CSV／日志／绘图、HINT 核心等价。
- 同次运行的六项官方 PSNR／SSIM 和像素损失测试通过；官方零方差 SSIM 测试产生一条运行时警告，断言通过。
- HINT 等价测试使用两份完整结构的同初始化权重，连续处理两张输入，从空 cache 开始；FP32 输出和 cache 满足 rtol=1e-5、atol=1e-6，无预训练权重。
- 一次性比较测试列表前两条真实 sequence 的全部 18 帧：KAIR 与 BasicSR uint8 PSNR 最大绝对差为 `3.552713678800501e-15`，SSIM 差为 0。
- 两种完整网络都在 GPU 0、1、总 batch 6、128×128 输入上完成连续训练，随后官方验证处理一条完整 sequence 的九张 512×512 图像，再继续训练。
- 原生网络权重、Adam 全部状态、scheduler 状态恢复检查通过；恢复后继续更新参数，loss 有限。测试用较小 milestone 实际跨过调度边界。

| 模型 | GPU 0 训练峰值已分配显存 | GPU 1 训练峰值已分配显存 |
| --- | --- | --- |
| SwinIR | 17.56 GiB | 17.41 GiB |
| HINT | 9.22 GiB | 9.02 GiB |

以上是 `torch.cuda.max_memory_allocated`，不包含缓存预留和驱动开销。短流程用于验证可运行性，不作为恢复质量或收敛结论。

两种模型还分别通过实际命令行检查：官方 `basicsr/train.py` 运行两步并保存，指定原生 `resume_state` 续训至第三步，再由 `scripts/eval_sar.py` 读取生成的 BasicSR 权重评估九帧。结果只包含三份 CSV，汇总日志只有一条。该检查采用临时配置与临时输出目录，不修改正式 YAML，不留下实验权重到新仓库。

`git diff --check` 通过。运行命令与操作说明见 [README_SAR.md](README_SAR.md)。未启动正式 220 万步训练。
