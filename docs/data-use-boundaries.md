# 数据用途与再分发边界

本项目限定于非商业科研，原始数据及派生数组不上传公开 GitHub；共享文件服务采用项目认证和最小成员访问。可获取作者处理包不等于其所有组成数据都获得统一的再分发许可证。

| 组成 | 已核实的一手来源 | 使用／再分发证据与缺口 |
|---|---|---|
| LRA 任务定义、处理代码 | [LRA 固定源码](https://github.com/google-research/long-range-arena/tree/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8)、仓库 Apache-2.0 LICENSE | 代码许可证不能替代各数据源权利。 |
| MEGA 处理包 | [作者配置页与下载链接](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md) | 下载源已固定并重建；没有找到覆盖全部数据的单一授权声明。 |
| ListOps | LRA 源码与原始任务文献 | 合成任务；仍需核对生成器来源和生成数据的声明，不能仅从主仓库 LICENSE 推断。 |
| IMDb | Stanford Large Movie Review Dataset 作者页 | 本次服务器请求 connection refused，未得到用途正文；记录失败，继续核实正式条款。 |
| AAN Retrieval | LRA／MEGA 提供的 AAN 派生任务 | 尚未取得与这个历史 AAN 版本严格对应的版权和再分发声明；ACL Anthology 通用版权页请求 404，即使取到通用许可也不能直接替代 AAN 版本证据。 |
| CIFAR-10 灰度序列 | [CIFAR 作者页](https://www.cs.toronto.edu/~kriz/cifar.html) | 作者公开下载并要求引用技术报告；页面未提供足以覆盖本项目再分发的统一许可，保留引用和数据来源。 |
| Pathfinder32 | LRA／MEGA 提供的图像和标签 | 需核对生成器、原始任务来源及发布声明；不将代码许可自动外推到图像。 |

当前结论：来源身份与现有授权下载已核实，逐任务许可和合作者再分发边界**仍待补证**。这不是法律判定，也不是停止已授权的私有科研处理；它使当前资格记录保持 `insufficient`，避免公开发布一个并不存在的统一开放数据许可证。获取记录详见 `docs/evidence/baseline-primary-sources-20261028.json`。
