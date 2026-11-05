# 五任务数据重建协议：mega-independent-v1

本轮以用户明确要求“没有副本，重建”及提供的两个公开来源为授权依据。目的为恢复可审计数据和检查输入、标签、划分；没有启动模型训练。规则在本轮转换代码中确定，不按模型得分选择划分。

## 来源选择与预算

- 采用 MEGA 作者仓库链接的 [lra.zip](https://dl.fbaipublicfiles.com/mega/data/lra.zip)，公开可直接获取，1,447,223,076 bytes。作者文档固定在 [aeaa4b4](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md)。本轮观测 SHA-256 为 `b141aacd5537ad6108862117597c41d9d2a67d6420c54f74c76df615073fee13`；不是声称作者发布了校验和。
- 用户提供的 [Kaggle 镜像](https://www.kaggle.com/datasets/a24998667/long-range-arena-processed)版本 1 文件清单含五任务及额外 Pathfinder128。API 标示总文件大小 22,281,022,099 bytes、CC0，描述为空；这些元数据不能单独证明上游许可或转换忠实性。本轮不下载其全部 pickle。
- [LRA 官方 README](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/README.md)给出的 GCS 整包本轮返回 403，停止该来源访问。MEGA 公开下载成功；未绕过访问限制。
- 只处理 ListOps、Text/IMDB、Retrieval/AAN、CIFAR-10、Pathfinder32；不展开 Path-X。下载约 1.45 GB，重建启动要求至少 8 GiB 可用空间，临时数组落磁盘、流式读写；CPU 执行，无 GPU 训练。原始包永久保留，旧输出不覆盖。

## 任务与表示

目标为各固定任务的样本分类，指标为测试准确率。样本是表达式、评论、文档对或图像；不以此声称 OOD 泛化、量子优势或新方法收益。正式模型选择仅允许查看训练集与独立验证集；测试标签只供数据审计和最终固定模型评估。

| 任务 | 输入 | 目标 | 长度与处理 |
|---|---|---|---|
| ListOps | MEGA 表达式 token ID | 原始 0–9 标签 | 截断/右填充到 2048，保留 MEGA 输入字典 |
| Text | MEGA IMDb 字节/字符字典 ID | 原始 0/1 标签 | 4000，保留 EOS，pad=1；与旧项目4096协议区分 |
| Retrieval | 两篇文档的独立 token 序列 | 字典解码后的原始 0/1 | 每篇4000；不能按旧转换器缩成2000×2 |
| Image | CIFAR-10灰度像素 | 0–9类别 | 1024个uint8像素，保留0–255；归一化留给共同运行协议 |
| Pathfinder | 实际Pathfinder32图像 | 路径连通二分类 | 1024个uint8像素；不使用随机标签占位数据 |

二进制索引按固定 MEGA 源码的 magic、版本、dtype、计数和指针解析，不固定假设 int16。输入词典随数据存档，不重新拟合词表。标签 ID 通过 `label-bin/dict.txt` 解码为语义标签，而非假定频率排序等于类别数字。四个特殊 token 由固定 Dictionary 实现确认；源数据和代码哈希记录于构建清单。

该数据只是可追溯输入；不能把当前将 token ID 直接当连续标量的旧训练实现称为忠实文本基线。后续共同模型协议须明确 embedding、mask、pooling、成对建模与像素归一化。

## 独立划分

ListOps、Retrieval、Image、Pathfinder保留包中的train/valid/test顺序映射。原始来源的独立性仍需全量审计，保留来源行号/长度元数据以支持重放。Retrieval按无序文档对查完全重复；共享单篇文档、近重复和真实文档ID仍是单独的研究边界，不声称验证了“未见过的文档”泛化。

Text原包valid和test的输入二进制逐字节相同，原方案不能用作独立模型选择。重建时：

1. 保留原test作为最终测试集，原valid不再用于模型选择。
2. 仅从原train建立新train/val。先按4000截断后的输入身份分组；与test输入身份重复的原train记录剔除，并记录数量，不借此读取测试标签或模型得分。
3. 同组冲突训练标签剔除并计数。组内所有相同评论保持同一划分。
4. 按 `SHA256(qkan-text-split-v1: + input_identity)` 排序，各类别依次选组进入验证，目标每类2500条；剩余组训练。分组可能使实际验证数略超目标；必须报告实际数。
5. 保存来源行号、保留长度、各文件哈希和剔除数量。原始包不修改。

这是独立验证的**适配协议**，不能将原LRA/MEGA论文数字或历史项目初测直接放进同协议比较。

## 2026-10-28：独立清洗版本 mega-independent-pairclean-v2

统一调度器在 `owner-recovery-3080ti` 成功执行固定 `ac8e0d4` 的 CPU-only 清洗审计任务。`curate_lra_pairs.py` 先核验 v1 文件哈希，在新的同级版本中复制四个未修改任务，仅重建 Retrieval：按截断后编码文档的无序对分组，剔除全部93个标签冲突组（按有序对计算为48组），余下组按 test→val→train 优先级保留一个来源行。选择规则不使用模型预测或性能，但数据构建审计访问了各划分标签；不能声称保留了完全未动的官方测试集。

Retrieval 保留 train 146731、val 18068、test 17419；分别剔除154/19/13条冲突组记录和201/3/5条重复或重叠记录。输出来源行号清单可从 v2 回溯到 v1，再回溯到原包。四个其他任务文件哈希不变。完整结构审计确认五任务所有跨划分编码级重叠和输入标签冲突均为0；共享单文档、近重复、数据用途与可学习性仍需单独证据。

服务器输出为 `/home/qkan-mamba-lra/data/versions/mega-independent-pairclean-v2`。紧凑清单、审计和资格记录在 `docs/evidence/mega-independent-pairclean-v2-*.json`，完成与哈希验收回执在 `docs/evidence/scheduler-completion-20261028.json`。这是适配后的研究数据候选版本，资格为 `insufficient`，未解锁正式比较。

## 资格门槛与交付

`audit_lra_data.py` 遍历声明范围的全部NPZ数组，检查成员/shape/dtype/标签/有限值、样本数与类别分布、长度一致性、文件哈希、跨划分完全重复和冲突标签。不同数值dtype与(N,L,1)/(N,L)用统一输入身份比对；Retrieval文档交换也计作同一对。临时SQLite记录输入哈希，汇总后销毁；不公开样本或逐样本哈希。

通过数值检查只得到 `insufficient`，缺文件也为 `insufficient`，明确缺陷为 `revise`；程序不自动生成qualified证书。正式比较还需：数据访问与用途边界、任务适配、来源/转换审计、文本输入语义、划分/关联单元与近重复边界、共同评估协议和必要可学习性证据。质量状态不依赖QKAN是否胜过基线。

大文件留在服务器认证共享空间，小清单与审计通过GitHub返回。数据重建、资格通过、实验包可运行和模型结果必须分别记录。
