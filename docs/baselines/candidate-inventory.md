# Baseline 候选与复现边界

2026-10-28：这是数据资格完成前的候选调查，不是已冻结队列，也不是复现成功报告。来源获取回执为 `docs/evidence/baseline-primary-sources-20261028.json`；正文缓存在服务器私有数据目录。该次收集有 22 个请求、20 个成功、2 个失败，没有浏览器升级或绕过访问限制。

| 候选 | 在本研究中的作用 | 一手依据与适配问题 | 当前结论 |
|---|---|---|---|
| Majority／固定线性参考 | 空模型、数据诊断下限 | 不承担长程机制建模；只用于资格诊断 | 已通过调度测试并完成诊断；不计正式 baseline |
| Transformer | 经典注意力参照 | LRA 原始论文和源码；必须冻结位置编码、padding、文档配对方式以及长序列资源预算 | 待机制清单与适配验证 |
| S4／S4D | 经典状态空间参照 | [作者仓库](https://github.com/state-spaces/s4/tree/e757cef57d89e448c413de7325ed5601aceaac13)含独立模块；SSM 参数专用学习率/weight decay 规则需保留。S4 与 S4D 是两个不同方法 | 优先审计官方模块；禁止用现有简化 `s4.py` 冒充 |
| MEGA | 与当前数据来源直接匹配的强注意力参照 | [作者 LRA 配置](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md)；fairseq 环境、EMA/gated attention、chunk 模式与任务协议需分别核对 | 第一批强候选；尚未运行 |
| S5 | 有完整 LRA 作者实验配置的强 SSM 参照 | [作者源码](https://github.com/lindermanlab/S5/tree/3c18fdb6b06414da35e77b94b9cd855f6a95ef17)，[论文](https://arxiv.org/abs/2208.04933)；JAX，MIMO SSM 和 parallel scan，不可简写为另一个 S4D | 第一批强候选；需独立环境与恢复适配 |
| LRU | 检验经典线性递归是否足够的参照 | [论文](https://arxiv.org/abs/2303.06349)；取得的 [minimal-LRU](https://github.com/NicolasZucchet/minimal-LRU/tree/bbd71b1104c399ec0d13819527d791949e66f599)明确是**非官方实现**，并说明部分 LRA 任务尚未测试、彩色 CIFAR 与 LRA 灰度任务不同 | 不能按官方五任务复现纳入；继续寻找作者工件或标记 hybrid/paper |
| Mamba-only | QKAN 的直接骨干对照 | [作者源码固定版本](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa)；仓库同时有 Mamba、Mamba2、Mamba3，必须明确架构。历史项目用 Mamba1，不能自动换版本 | 必需的骨干对照；不因语言建模成绩宣称 LRA SOTA |
| Identity／参数量匹配 MLP／Fourier 或 KAN gate | 以后分析 QKAN 增益来自哪里 | 同骨干、同 head、同训练预算；identity 应与 Mamba-only 数值等价；参数量和计算量分别报告 | 方法实验阶段的必要控制，当前不执行 |

2024–2026 年补充初筛见 `recent-screen.md`：已取得两页 arXiv 精确短语搜索和六篇任务相关论文，B₂S₆、强 Transformer 配方及近期门控模型进入源码追踪。正式冻结前仍须完成条件比较和代码审计，按预先声明的性能、代码可验证性与任务覆盖标准选出强方法 top-x，记录每个排除原因。此处列出 MEGA/S5 不代表已证明它们是当前前 x 名。近邻任务、彩色 CIFAR、Path-X 和本项目五任务均值不可混写。

取得的源码 pin：LRA `cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8`；MEGA `aeaa4b44592cd1d60a9a34554e359eda2a62b03b`；S4 `e757cef57d89e448c413de7325ed5601aceaac13`；S5 `3c18fdb6b06414da35e77b94b9cd855f6a95ef17`；Mamba `e9594ce1c732d97440f0332fdc43170a2294dbfa`。固定 commit 只证明来源身份，不证明运行保真。

当前所有模型的 fidelity 状态均为 `not_verifiable`。尚无新的正式 baseline 分数，门户不得把数据诊断计入模型结果。
