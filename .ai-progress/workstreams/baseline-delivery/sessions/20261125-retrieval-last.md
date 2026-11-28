# Retrieval 统一后置

- 用户要求：所有 baseline 模型先做完 ListOps、Text、Image、Pathfinder，所有未完成 Retrieval 统一后置。
- 交付：统一入口 `resume --task-order retrieval-last`；正式候选 Suite 默认采用该顺序，普通历史 admitted Suite 兼容原行为。`manifest` 显式恢复原排序。看板显示相同全局顺序。
- 稳定核心：只改 scripts/experiment.py 和 scripts/formal_progress.py；在原固定执行目录获取 .bootstrap 控制脚本即可。原 Suite 清单哈希、科学绑定、配置、训练/保存/恢复函数、完成结果与 Retrieval 断点不变。
- 适配机制：私有派发函数命名空间仅对本 Suite 清单读取提供稳定排序；原校验函数保留原磁盘视图；不修改共享模块全局变量。task-order.json 与原清单 SHA-256 绑定，持有监督器/Suite 锁时更新。
- 验证：QT2 已有环境，标准库 unittest tests.test_baseline_experiment_entry，34 项通过；包含实际 Mamba 父派发函数选择非 Retrieval、暂缓 Retrieval 状态保留、全局 25 项稳定分组及原字节不变、看板队列一致、原安全暂停/重复进程锁/只读观察兼容。没有安装依赖或执行框架/GPU/正式训练。
- 操作：若仍训练，原终端 Ctrl-C 一次等待保存退出，再执行 docs/baselines/experiment.md 顶部整段；目标 original89353fc / mamba-fused-formal-20261124T061405Z-b57105a1。没有实际远程切换或启动证明，不将代码交付当作训练已开始。
