# 2024–2026 年 LRA 一手文献初筛

检索日期 2026-10-28。arXiv 精确短语 `"long range arena"`，按首次公告时间倒序、每页 50 篇，取得前两页，并沿其中六篇的 arXiv HTML 阅读任务表、机制和代码链接。获取回执是 `baseline-source-supplement-20261028.json` 与 `baseline-recent-details-20261028.json`。API 失败保留为失败；HTML 搜索正常。检索按论文首次公告排序，结果页的 Submitted 行可能反映后续版本日期，不能混作首次发表日期。

这是有边界的初筛：未覆盖不含该精确短语的论文、全部会议库和量子/KAN 门控文献，因此不是完整系统综述或全球 SOTA 排名。

| 文献 | 对本项目的直接意义 | 纳入／暂缓依据 |
|---|---|---|
| [Block-Biased Mamba for Long-Range Sequence Processing](https://arxiv.org/html/2505.09022v1) | 分析 Mamba 在 LRA 的表达能力、归纳偏置和稳定性，提出 B₂S₆；直接挑战“在弱 Mamba 上加一个 gate 就足够”的论证 | 优先源码追踪与机制审计。论文给出六任务对照，含 Path-X；其均值不能与本项目五任务均值混用。未取得作者可执行仓库前保持 not_verifiable |
| [SeRpEnt](https://arxiv.org/html/2501.11729v1) | 通过选择性重采样压缩序列，与 QKAN 门控的潜在作用需要区分 | 机制相关候选；需继续核对任务完整性、重采样额外计算和作者代码 |
| [S7](https://arxiv.org/html/2410.03464v1) | 有输入选择性、稳定重参数化和完整 LRA 表；部分任务强、图像类弱，不能只取有利任务 | 保留任务级比较。论文自己的六任务表并未表明整体优于 MEGA/S5，不列为当然最强参照 |
| [On the locality bias and results in the Long Range Arena](https://arxiv.org/html/2501.14850v1) | 提醒 LRA 改进可能来自局部归纳偏置和训练技巧；给出可竞争的 Transformer | 必须影响 claim 和消融设计；加入局部/顺序破坏诊断与强训练配方的比较计划。[作者代码](https://github.com/pablomiralles22/paper-LRA-source)待固定版本和验证 |
| [FLARE++](https://arxiv.org/html/2608.11519v1) | 2026 年新注意力，确有五任务 LRA 表，但主要应用是 PDE surrogate | 近期覆盖记录，不能只凭更新日期取代已充分验证的强 LRA 方法；低秩动态路由与本研究具体机制关联较弱 |
| [Naju](https://arxiv.org/html/2607.21000v1) | 2026 年独立 retention/write gate，与门控研究相关；给出同预算 Mamba/Mamba2 对照 | 有用的机制/预算候选；作者明确将自己的同预算结果与配置不同的既有论文分数分区，沿用此可比性边界。作者代码与状态预算待审计 |

由此调整候选优先级：先保证 Mamba1 骨干、正确的官方 S4D、MEGA/S5 和有竞争力的 Transformer；把 B₂S₆ 与近期门控模型列为必须处理的近邻，不得只比较历史弱骨干。最终 top-x 仍须按任务覆盖、相同预算可比性、来源可验证性冻结；若重要候选没有可验证源码，必须列出缺口和论文复现成本，不静默删除。

LRA 的分数只支持所测分类任务的效用。五任务平均提升不能单独证明通用长程记忆、量子优势或纠缠收益；后续需要局部窗口、序列顺序、长度外推及参数／计算匹配证据。当前没有进行这些方法实验。
