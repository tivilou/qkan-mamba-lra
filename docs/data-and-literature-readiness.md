# 数据与相关工作准备状态

当前结论（2026-10-28）：任务已选择 LRA 的 ListOps、Text/IMDB、Image/CIFAR-10、Pathfinder、Retrieval。五任务已从固定 MEGA 公开包重建，但资格状态仍为 **insufficient/revise**，尚未解锁正式基线比较。

## 数据

旧实例已释放且无合作者副本。用户提供的 MEGA 作者链接已在替代服务器下载并固定：1,447,223,076 bytes，观测 SHA-256 `b141aacd5537ad6108862117597c41d9d2a67d6420c54f74c76df615073fee13`。Kaggle version 1 只用于文件清单与来源交叉核对，没有下载其 22.28 GB pickle。原始包和重建输出均不覆盖。

`mega-independent-v1` 已生成五任务 train/val/test、来源行号、字典、shape/dtype、类别分布和文件哈希。Text 原 valid 与 test 输入完全相同，因此改为仅从原 train 按编码身份和类别构造新验证集，保留原 test；长度固定为 4000，这与旧项目 4096 协议不可混用。Retrieval 每篇文档长度为 4000，不使用旧转换器的 2000×2。

首轮完整数组审计发现 ListOps、Text、Image、Pathfinder 没有编码级跨划分重复或冲突；Retrieval 有 train/val 35、train/test 46、val/test 1 个无序文档对重叠，以及 93 个无序文档对冲突标签组，因此源构建状态为 `revise`。`lra-pair-qualification-v1` 将在独立版本中剔除冲突组并按 test→val→train 保留唯一对；统一调度任务 `d18c07ab...` 已安全排队，尚未产生 attempt 或结果。

即使该结构审计通过，程序也只会给出 `insufficient`，不会自动签发 qualified。仍需逐任务数据许可/用途边界、Retrieval 共享单文档与近重复边界、冻结输入变换和必要可学习性证据。MEGA/LRA 代码许可或 Kaggle 页面 CC0 标签不能代替这些数据级证明。

## 相关工作范围（尚未完成调研）

1. LRA 官方任务、数据处理与评估协议，以及后续复用中的差异。
2. 长序列分类的状态空间方法：S4/S4D、Mamba，以及任务匹配的近期强方法。
3. Transformer 与简单经典参考，输入表示、长序列训练、参数和计算预算。
4. KAN、三角/Fourier 非线性、门控与参数匹配对照。
5. 与同类任务相关的量子门控/量子机器学习方法；区分模拟器与硬件、独立单比特与纠缠、任务收益与量子优势。

文献清单需包含论文/官方代码、版本、任务/划分、指标、计算预算、复现条件及可比性。没有核实的近期排名不写为 SOTA。不是把所有论文都跑一遍，而是据证据冻结可辩护的队列。

## 基线状态

当前上游 Mamba-only 初测尚未形成正式忠实性证书；S4 为简化自写实现；Transformer 已有代码但无已恢复结果。正式队列尚未选定。历史文档中的平均准确率和 >83% 目标未附核实来源，不能作正式比较依据。

下一步：等待统一调度任务在 128 MB worker headroom 恢复后自动执行，收回 build、pair-curation、diagnostics、qualification 四个紧凑记录并核验哈希；同时完成数据许可与 Retrieval 关系单元审计。相关工作调研可以并行。数据 qualified、队列与协议冻结、实现忠实性和运行预检通过后，才发放正式基线任务。
