# 数据用途与再分发边界

本项目限定于非商业科研，原始数据及派生数组不上传公开 GitHub；共享文件服务采用项目认证和最小成员访问。可获取作者处理包不等于其所有组成数据都获得统一的再分发许可证。

| 组成 | 已核实的一手来源 | 使用／再分发证据与缺口 |
|---|---|---|
| LRA 任务定义、处理代码 | [LRA 固定源码](https://github.com/google-research/long-range-arena/tree/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8)、仓库 Apache-2.0 LICENSE | 代码许可证不能替代各数据源权利。 |
| MEGA 处理包 | [作者配置页与下载链接](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md) | 下载源已固定并重建；没有找到覆盖全部数据的单一授权声明。 |
| ListOps | [LRA 固定生成器](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/data/listops.py)与原始任务文献 | 已取得生成代码及 Apache-2.0 头部，LRA README 指向作者发布的 TSV；这能固定来源和生成方式，不能把代码许可自动当作生成数据的单独授权。保留引用，当前不再分发数组。 |
| IMDb | [原作者论文 §4.3.2、§5](https://aclanthology.org/P11-1015.pdf)；Stanford Large Movie Review Dataset 数据卡 | 2026-11-04 取得作者论文原文，明确把 50,000 条评论作为未来研究基准向公众发布。该研究发布意图支持本项目限定的内部非商业研究；不据此声称商业使用或任意再分发权。作者网站 404／连接失败如实保留，不再以网站不可访问等同于缺少所有用途证据。引用 Maas 等 2011；不再分发文本／数组。 |
| AAN Retrieval | LRA／MEGA 提供的 AAN 派生任务；[AAN 作者下载与用途页](https://clair.eecs.umich.edu/aan/download.php) | 2026-11-04 取得明确条款：仅研究、禁止再分发、发表需引用作者并告知。作者下载流程要求姓名／机构／邮箱登记；本次仅读取页面，没有提交表单，也不声称已有登记凭证。LRA 同样禁止再分发。原文和派生 token 数组均不进入我们的共享服务；交付来源、重建代码和哈希，保留实际作者获取链路记录。 |
| CIFAR-10 灰度序列 | [CIFAR 作者页](https://www.cs.toronto.edu/~kriz/cifar.html) | 作者公开下载并要求引用技术报告；页面未提供足以覆盖本项目再分发的统一许可，保留引用和数据来源。 |
| Pathfinder32 | [LRA 参数与 builder](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/data/pathfinder.py)、MEGA 转换器；[生成器 MIT 许可](https://github.com/drewlinsley/pathfinder/blob/7b538865667dfe16f949ed3b7c8dc6a65ca87099/LICENSE) | 来源、hard 子任务、seed=42 的 80/10/10 处理和全部像素／标签一致性已核实。固定生成器代码为 MIT；当前版本不是历史生成 revision 的证明，代码许可不自动外推到处理包全部图像。当前不再分发数组。 |

当前结论：逐任务内部非商业研究范围评审通过，数据资格为 `qualified`；机器评审见 `docs/evidence/task-internal-use-review-20261104.json`。评审接受原作者明确的研究发布意图、固定作者处理包获取链及严格的不再分发边界，不要求统一开放许可证。代码许可不外推为数据许可，禁止再分发不解释为禁止内部研究；不声称已完成未有凭证的 AAN 原始下载登记。若另外获取 AAN 原始包，应遵守作者登记流程。范围、来源或用途改变时重新评审；模型正式发布仍未通过。

2026-10-28 补证回执为 `qualification-sources-20261028.json`、`qualification-use-sources-20261028.json`、`qualification-followup-sources-20261028.json`、`qualification-author-sources-20261028.json`。成功内容以 SHA-256 固定，缓存命中保留原获取时间且明确标记未重新验证；失败不会填成许可结论。当前文件交换仅承载允许的代码、配置、权重、compact 摘要及经审查的数值证据；上述数据数组和原文均暂不经本项目服务再分发。Case Study 的文本样本也需遵守同一任务边界。
