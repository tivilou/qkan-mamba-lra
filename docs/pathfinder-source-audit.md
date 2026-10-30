# Pathfinder 来源与输入保真核验

2026-10-28：**Pathfinder 的来源／可学习性证据缺口已补齐；整体数据资格仍为 insufficient。** 本次没有训练新的模型，没有获得本研究的 Pathfinder baseline 分数。

## 来源链与外部参考

[固定 MEGA 配置页](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/README.lra.md)直接发布了本项目取得的 `lra.zip`。[固定转换器](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/examples/mega/lra/data_process.py)选择 `curv_contour_length_14`，以 seed=42 将 200,000 张图划为 160,000/20,000/20,000；图像转灰度后按行展开，标签读取 metadata 第四列。

[LRA builder](https://github.com/google-research/long-range-arena/blob/cd31e5c6b8e5bceabd28de2d2afb23f7ae5d36d8/lra_benchmarks/data/pathfinder.py)把这一目录定义为 hard，记录了 32×32、contour_length=14 等生成参数。[MEGA 论文 Table 2](https://arxiv.org/html/2209.10655v2#S4.T2)报告 Pathfinder accuracy 96.01%，MEGA-chunk 为 94.41%。这些是外部作者报告，只用于确认这个作者发布任务具有已报告的可学习信号，不进入本项目正式结果表，也不作为本项目性能目标或复现成功证据。

Pathfinder 生成器当前可核验版本为 `7b538865667dfe16f949ed3b7c8dc6a65ca87099`，附 MIT 代码许可。它不是 MEGA 当年实际生成数据所用 revision 的证明；作者处理包的固定字节身份是本研究的数据版本锚点。

## 逐行核验

统一调度 CPU 任务 `cc37d99eae4146629545830ede9ec554` 在代码 `52257f298ed591983a2345dfeedc9f5f0b2f11a1` 执行 `scripts/audit_pixel_source.py`：先核验原包 SHA-256，再流式比较每条源文本行与 NPZ 的全部像素及标签，包括 train、validation、test。没有训练、预测或依据 test 调参。

| 任务 | train / val / test | 像素不一致行 | 标签不一致行 |
|---|---|---:|---:|
| Image | 45,000 / 5,000 / 10,000 | 0 | 0 |
| Pathfinder32 | 160,000 / 20,000 / 20,000 | 0 | 0 |

终止回执为 complete、returncode=0、quiescent=true；本地下载报告的 SHA-256 与终止回执一致。完整报告见 `docs/evidence/mega-document-v3-pixel-source.json`，验收见 `docs/evidence/pixel-source-scheduler-completion-20261028.json`。

## 输入缩放及结论上限

[MEGA PixelSequenceDataset](https://github.com/facebookresearch/mega/blob/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/fairseq/data/pixel_sequence_dataset.py)使用 `(x/255−mean)/std`。Pathfinder 配方为 mean=std=0.5，等于本研究公共输入 `x/255−0.5` 的 2 倍。对全部 256 个 uint8 值的 float32 核验表明，公共输入可恢复原始像素，且作者输入严格等于公共输入乘 2。

这支持一个有限推论：任务的信息与标签保留，已有作者结果和严格数据等价共同支持可学习性，不必把弱线性参考未学会当作数据失败。它不证明两种缩放下的优化轨迹、超参数或实际性能相同。正式 MEGA 复现要在声明的适配边界恢复作者缩放并测试；公共协议比较要单独声明共同缩放，禁止二者混报。Image 的作者配方 mean=0.48/std=0.24 也需在相应适配器中记录。

## 资源与测试

已授权替代 3080Ti 的注册 CPU 节点运行；没有 GPU。审计预算为 1 CPU、128 MiB 申报 RAM、32 MiB 输出磁盘、30 分钟上限、最多一次尝试。数据只读、不新建数据版本。新增测试覆盖单像素／标签损坏、源行数差异、像素／标签越界及全部像素缩放；连同已有测试，共 31 项在调度 preflight 中通过。
