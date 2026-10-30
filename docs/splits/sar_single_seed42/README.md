# 单帧 SAR 数据划分

原 H/L 数据版本：`dataset_range_2dsft/range_2d_sft_qh2p5_ql1p5_s1_fr0p9_fa0p5_phi0_lp0p99_hp99p9_roi600_eb64`。全 Q=2 版本 `dataset_range_2dsft/range_2d_sft_q2_s0_fr1p3_fa0p4_phi0_lp0p99_hp99p9_roi600` 共用这些 sequence 列表。

原划分将全部 4098 个 LQ sequence 目录名按字典序排序，用 Python `random.Random(42).shuffle` 打乱，前 3688 条用于训练，剩余 410 条作为测试池。本次保持原训练列表逐行不变，从原 410 条列表中用 `random.Random(42).sample(names, 82)` 固定选出验证 sequence；验证、最终测试各自保留原列表中的相对顺序。

| 文件 | Sequence 数 | 帧数 | 用途 |
| --- | ---: | ---: | --- |
| train.txt | 3688 | 33192 | 训练 |
| val.txt | 82 | 738 | 训练期间验证 |
| test.txt | 328 | 2952 | 训练完成后的最终测试 |

三个集合互不重叠，并完整覆盖原始数据。每条 sequence 的全部九帧属于同一集合。SwinIR、HINT、UCAN 的原 H/L 和全 Q=2 配置共用本划分，最终测试集不用于中间检查点选择。当前训练参数与配置入口见 [README_SAR.md](../../README_SAR.md#数据和训练协议)。

split 路径相对于工作区根目录；数据根目录使用 YAML 中的机器绝对路径。迁移机器时更新数据根目录。
