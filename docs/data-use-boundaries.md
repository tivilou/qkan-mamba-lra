# 数据用途与再分发边界

本项目限定于非商业科研，原始数据及派生数组不上传公开 GitHub；共享文件服务采用项目认证和最小成员访问。可获取作者处理包不等于其所有组成数据都获得统一的再分发许可证。

| 组成 | 已核实的一手来源 | 使用／再分发证据与缺口 |
|---|---|---|
| LRA 任务定义、处理代码 | [LRA 固定源码](https://github.com/google-research/long-range-arena/tree/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8)、仓库 Apache-2.0 LICENSE | 代码许可证不能替代各数据源权利。 |
| MEGA 处理包 | [作者配置页与下载链接](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md) | 下载源已固定并重建；没有找到覆盖全部数据的单一授权声明。 |
| ListOps | [LRA 固定生成器](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/data/listops.py)与原始任务文献 | 已取得生成代码及 Apache-2.0 头部，LRA README 指向作者发布的 TSV；这能固定来源和生成方式，不能把代码许可自动当作生成数据的单独授权。保留引用，当前不再分发数组。 |
| IMDb | Stanford Large Movie Review Dataset 作者页；[Stanford 命名空间发布的数据卡](https://huggingface.co/datasets/stanfordnlp/imdb/blob/main/README.md) | 作者主页、当前作者域名和 Stanford 别名均未取得正文；保留连接失败，不持续重试。数据卡确认来源、25,000/25,000 划分和原论文，但 license 为 other，Licensing Information 待补，不能替代原始使用声明。当前不再分发文本／数组。 |
| AAN Retrieval | LRA／MEGA 提供的 AAN 派生任务 | LRA 固定 README 明确表示不能再分发此数据，仅发布文档对 ID。禁止将 AAN 文本或派生 token 数组上传我们的共享服务。后续取得了 [ACL FAQ](https://aclanthology.org/faq/#what-is-the-copyright-for-materials-in-the-acl-anthology) 与 [AAN 作者页](https://clair.eecs.umich.edu/aan/index.php)，但通用 ACL 许可不直接替代这个历史 AAN 版本的声明。向合作者交付来源、重建代码和哈希，独立核对其获取条件。 |
| CIFAR-10 灰度序列 | [CIFAR 作者页](https://www.cs.toronto.edu/~kriz/cifar.html) | 作者公开下载并要求引用技术报告；页面未提供足以覆盖本项目再分发的统一许可，保留引用和数据来源。 |
| Pathfinder32 | [LRA 参数与 builder](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/data/pathfinder.py)、MEGA 转换器；[生成器 MIT 许可](https://github.com/drewlinsley/pathfinder/blob/7b538865667dfe16f949ed3b7c8dc6a65ca87099/LICENSE) | 来源、hard 子任务、seed=42 的 80/10/10 处理和全部像素／标签一致性已核实。固定生成器代码为 MIT；当前版本不是历史生成 revision 的证明，代码许可不自动外推到处理包全部图像。当前不再分发数组。 |

当前结论：来源身份与现有授权下载已核实，逐任务许可和合作者再分发边界**仍待补证**。这不是法律判定，也不是停止已授权的私有科研处理；它使当前资格记录保持 `insufficient`，避免公开发布一个并不存在的统一开放数据许可证。获取记录详见 `docs/evidence/baseline-primary-sources-20261028.json`。

2026-10-28 补证回执为 `qualification-sources-20261028.json`、`qualification-use-sources-20261028.json`、`qualification-followup-sources-20261028.json`、`qualification-author-sources-20261028.json`。成功内容以 SHA-256 固定，缓存命中保留原获取时间且明确标记未重新验证；失败不会填成许可结论。当前文件交换仅承载允许的代码、配置、权重、compact 摘要及经审查的数值证据；上述数据数组和原文均暂不经本项目服务再分发。Case Study 的文本样本也需遵守同一任务边界。
