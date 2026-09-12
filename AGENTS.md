# SAR_Sequence_Restore 协作规则

本项目基于标准 BasicSR，开展单比特 SAR 单帧基线与后续序列恢复研究。

- 使用最小修改，遵循 YAGNI；只在输入、文件等真实边界验证，不增加无需求的 fallback、hash、wrapper 或兼容层。
- 当前 SwinIR 使用官方 SwinIRModel，HINT 使用官方 SRModel，训练采用原生 DataParallel。
- 新网络通过 ARCH_REGISTRY、新数据集通过 DATASET_REGISTRY 注册；未来确有特殊训练行为时才使用 MODEL_REGISTRY 扩展。
- 不修改通用训练、优化器、调度、日志和检查点机制来模仿旧 KAIR 行为。
- SAR sequence 级 split、九帧展开、灰度、共同裁剪与 metadata 是必须保留的数据协议。
- HINT qv_cache 与核心算法按来源保留；算法研究应作为单独任务。
- 文件使用 UTF-8；关键步骤、变量和设计意图使用简洁中文注释。
- 新文档放 docs/，保留官方原有文档和许可证布局。
- 仅在明确要求时编写或修改代码；直接使用当前工作区，不使用 git worktree。
- 完成代码或文档修改并验证后，优先用明确的中文说明 git commit。
- 三个来源目录仅用于迁移参考，不修改；原始数据只读，运行不依赖来源代码。
