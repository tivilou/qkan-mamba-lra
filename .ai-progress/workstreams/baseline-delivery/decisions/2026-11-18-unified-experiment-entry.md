# 统一用户实验入口与固定执行器分离

- Status: active
- Date: 2026-11-18
- Basis: 用户要求训练和原美化看板一体；确认整理多个一键入口，并要求最终交付当前速度优化版的一体化续跑命令。

## Decision

日常推荐统一`scripts/experiment.py`的start/resume/watch/upload/diagnose操作。训练／续跑默认在同一终端显示原Dashboard／Terminal；只读观察是明确的watch操作，不代替真实训练启动。历史Shell继续兼容；先统一操作、文档与分派，再在实际引用／绑定核查后考虑物理删除。

统一控制器不拥有科学状态，不改变准入、策略、模型、预算、样本时钟或检查点合同。已有Suite仍在原固定干净checkout中，通过原解释器／执行模块恢复最新完整状态。新控制器与原UI可放入忽略的`.bootstrap`目录，避免对已绑定科学文件执行更新。resume不自动迁移或重复已有诊断。

## Evidence

- docs/baselines/experiment.md
- docs/baselines/experiment-entrypoints.json
- docs/evidence/unified-experiment-entry-delivery-20261118.json
- tests/test_baseline_experiment_entry.py
