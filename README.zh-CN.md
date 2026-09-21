# SAIP：摘要感知迭代伪标签生成

公开代码默认面向 **ActivityNet Captions**，完成原始视频 → BLIP 特征与字幕 → 候选生成 → 四维 SFS 选择 → 边界迭代校准 → 伪标签导出与质量评估。

生成阶段不使用目标视频的人工边界、字幕或类别；预训练模型本身包含外部监督知识。本仓库不实现 PDVC/Vid2Seq 等下游 DVC 训练。

请按 [完整复现步骤](docs/reproduce_activitynet.md) 安装环境、准备本地 BLIP 模型和视频，并运行 `prepare_activitynet.py`、`extract_blip.py`、`run_experiment.py` 与 `evaluate_pseudo_labels.py`。默认配置为 `configs/activitynet.yaml`，路径由命令行传入，无需修改源码。具体对应关系见 [论文与代码映射](docs/paper_mapping.md)。

输出 `train_pseudo.json` 使用秒级时间戳；`saip_report.json` 保存配置、逐轮日志与是否收敛；`evaluation.json` 保存逐视频及整体指标。评估包含一对一时序精确率/召回率/F1、覆盖率、平均最佳 IoU 和局部词重叠 F1。词重叠不等于字幕语义正确性，达到迭代上限也不等于收敛。

Charades 适配器仅用于本地 500 视频的代码运行验证，不作为论文新增实验，也不替代原文 ActivityNet 实验。公开版本保留可选适配器，主入口、命令示例和默认配置均面向 ActivityNet Captions。模型权重、视频、人工标注及本地运行缓存不包含在源码发布包中。

测试：`python -m pytest -q`。代码采用 MIT 许可证，模型与数据遵循各自许可。
