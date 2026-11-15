# 可开关训练优化：第一阶段交付

2026-11-15。第一阶段交付插件配置、尾部 PAD 裁剪和有界 train 数据统计。尚未接入正式训练；训练提速结论仍待原生验证。既有 Graph32 对 Graph16 的约 5.21% 吞吐收益属于此前独立诊断，不是本插件的结果。

已收到本次统计：前 4096 对 train 样本、micro32、128 长度对齐下，两篇文档的 128 个微批均仍执行宽度 4000；矩阵单元减少为 0%。按预设停止条件，跳过本配置的 PAD 正式训练接入，开关保持关闭。报告只覆盖约 3.48% 的固定 train 前缀，不能据此断言所有样本没有 PAD。结果见 `docs/evidence/padding-opportunity-review-20261115.json`；下面保留的统计命令已执行成功，本次无需重复。

## 开关与边界

| 配置项 | 本版能力 | 默认 | 后续条件 |
| --- | --- | --- | --- |
| `padding_trim` | 已实现 CPU token 边界及统计，可独立启用 | 关 | 真实裁剪空间、数值/RNG/恢复/trace、完整更新计时 |
| `bf16_amp` | 已登记，启用会明确报错 | 关 | 精度协议单列、离线环境、原生收敛及数值评估 |
| `flash_attention` | 已登记，依赖 BF16，启用会明确报错 | 关 | 离线固定内核、真实 kernel 证明、确定性反向验证 |
| `fp32_attention_tuned` | 已登记，启用会明确报错 | 关 | 新内核及其正确性/收益证据 |

显式注册表位于 `experiments/baseline_delivery/optimization_plugins.py`，没有动态扫描或静默降级。拼写错误、非法字段、依赖缺失和冲突均提前拒绝。`deterministic` 是拟要求的合约，不是原生认证结果。BF16 与 FP32 定制路线互斥。

两个预设位于 `experiments/baseline_delivery/optimization-presets/`：`baseline.json` 全关，`padding-trim.json` 只开裁剪。全关时适配器返回同一输入对象，不检查、不复制、不改变 batch。默认对齐长度为 128，可配置为 1–4000；该参数是序列长度对齐，不是 micro-batch 的步长。

裁剪分别对两篇文档计算当前微批中最后一个非 PAD（ID=1）的位置，向上对齐后截断共同尾部。保留行顺序、文档身份、内部 PAD、有效 token 的绝对位置。标签不进入插件接口。示例：长度 4000 的微批最大有效末位置为 1500，按 128 对齐后执行宽度为 1536；只估算注意力矩阵单元减少，不把它换算成 GPU 提速。

形状变化可能改变 dropout RNG 消耗、归约顺序、内核选择及 Graph 缓存。token 完整性不等于训练数值等价。正式 worker、迁移白名单、模型、FP32 协议和已完成结果均未接入此开关。不得把配置字段直接塞入旧正式任务冒充已通过准入。

## 交给师弟执行

在服务器的 Bash/tmux 终端完整执行下面这一组命令。它准备新版独立 checkout、复用账号已有私密配置，然后进行统计；无需打断正在训练的 Python 进程。它只读固定 train 数据，既不访问 GPU，也不读取 checkpoint；初次校验会完整读取 train 文件及其已提取缓存的哈希，可能产生短时磁盘竞争，可选 I/O 空闲时执行。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_PROFILE_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_PROFILE_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/profile_optional_padding.sh \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache \
  --optimizations experiments/baseline_delivery/optimization-presets/padding-trim.json \
  --micro-batch 32 --max-rows 4096
```

`prepare_adaptive_checkout.sh` 的末尾会打印通用 GPU readiness 示例；本次继续执行上面给定的 `profile_optional_padding.sh` 即可。原训练 checkout 的 HEAD 保持原版本，统计输出单独存放于新 checkout。

入口复用原 b07 Suite 的数据目录、原 train SHA 和现有数组缓存；缓存缺失直接报错，避免生成第二份大数组。只观察 train 前 4096 行及其连续微批，不读取 val/test，不按标签选样，不排序、不增添采样随机数，也不外推为整个数据集的长度分布。`--max-rows` 硬上限 16384；微批只接受当前关心的 16/32。需要对照时可单独换用全关预设；通常先运行一次裁剪预设即可，它已经检查全关路径的对象一致性。

账号已有私密 exchange 配置可复用；若新终端尚未载入，先按既有流程运行 `.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure`。凭据保持在私密配置中。

输出为 `runs/padding-profile-<时间>-<随机后缀>/padding-profile.json`。只自动回传这个聚合 JSON，命名空间为 `qkan-mamba/padding-profile/<目录名>`，校验 SHA-256 回执；失败保留已有 `artifact-delivery.json` 及本地文件，重试同一上传声明即可。无模型副本、checkpoint、token 明文或全量逐样本轨迹；报告只有长度分位数、128 桶计数、执行宽度计数、源/配置/插件哈希和 CPU 诊断耗时。

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/PADDING_PROFILE_DIRECTORY \
  --experiment padding-profile --file padding-profile.json
```

其中 `PADDING_PROFILE_DIRECTORY` 替换为本次输出目录名；重试无需重新统计或训练。

## 如何根据结果继续

先看 `measurement.attention_cells_reduction_fraction`、两篇文档的 `span_quantiles` 和 `execution_width_histogram`。减幅为零就终止这条 PAD 路线，优先研究注意力反向或精度路线；出现有意义减幅后，再准备绑定新机制 dossier 的 v2 候选计划，进行 GPU 诊断。前缀统计有抽样局限，接近零或高度波动时需补充分散位置的预先固定 train 样本规则，再决定。

下一阶段要求：先冻结选样和 trace.v2 契约，再证明同一样本的完整 token/位置 lineage、数值和 RNG 合约、独立进程恢复、精确回滚；最后与现有 Graph32 做相同逻辑 batch 的 AB/BA 完整更新计时，并计入 Graph 准备/重建及峰值显存。改变精度的 BF16 单列比较，不把它当 FP32 位级等价优化。

未来组合顺序为 token 预处理 → 精度上下文 → attention 后端 → Graph；每层独立关闭，逐项和组合消融。禁止只看显存占比判断收益；目标是完整训练更新的 samples/s、显存峰值和净耗时。此阶段架构计划保持 `approved`：尚无真实 sample-trace.v2 fixture、语义读取及重放证据，不标记 `implemented`。

## 本地验证范围

仅使用 QT2 标准库、数组替身和框架导入阻断测试，覆盖 null 身份、双文档边界、内部 PAD、对齐、非法配置、不可用选项、精确 train/hash 绑定、标签隔离、缓存缺失、聚合计数、实际 CLI 及成功/失败回传分支。真实 NumPy、Torch、GPU、精度及吞吐验证由协作者执行。本机测试不构成原生加速证据。
