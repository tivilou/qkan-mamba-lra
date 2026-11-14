# Retrieval 两层 Graph 候选的验证入口

本版实现隔离候选 `eager_partitioned`，每张 Graph 覆盖连续两层，目标是把物理 micro 从16提高到32。逻辑 batch 维持128，FP32、注意力实现、dropout、样本预算、优化器和学习率时钟保持原配置。代码与控制层检查已经完成；实际 CUDA 捕获、数值一致性、显存峰值和净提速须由协作者验证。

## 实现范围

- 每个文档有独立流与图池，池内按 F0、F1、B1、B0 捕获；第二段的隐藏输入梯度继续传回第一段。
- embedding、位置编码、末尾 LayerNorm 和池化各执行一次，原始模型与 checkpoint 参数名保持一致。
- 分段路线每次完整更新同步一次副本，取代每个 micro 都同步。该副本优化与分层共同组成此次候选，暂不分别归因收益。
- 诊断强制 micro16/32；旧整图容量上限仅在该诊断控制器中解除。OOM、容量失败或回退均阻止该候选通过，避免把实际 micro16 报为32。
- 先分别执行 micro16、32 的新进程恢复和 trace 检查，再从最新 b07 完整断点测试 micro32 的完整逻辑128更新。
- 候选必须通过相同 micro 的普通执行数值/RNG检验，并且同时快于同 micro 普通执行及现有整图 Graph micro16。任意测得正收益均可通过，没有最低百分比门槛。
- Graph16/32 的 AB/BA 计时包含参数同步、梯度合并、优化器和更新边界等待；各次从同一快照开始。预热、重建和数值校验计时排除，单独记录准备峰值。测速时只驻留一路图池，另记稳定 allocated/reserved/free/peak。
- 两种 micro 的随机数与浮点轨迹可能不同，跨 micro 不声明逐位等价。各路线分别通过同 micro 的数值/RNG检查，恢复仍要求精确一致，最终恢复原始模型/Adam/调度器/RNG，样本与更新提交均为0。

当前普通训练入口仍使用已有 `eager_parallel`；新分层候选仅允许隔离诊断。micro20/24/28、步长4、多形状尾批缓存和新的动态升档器不在本版实现范围内，须先依据32的原生结果决定是否需要。

## 使用条件与指令

运行中的训练继续使用 b07。准备代码可以随时做；下面的 GPU 诊断需等当前训练按既有流程暂停、完整断点保存并退出后再运行，或另选空闲 GPU。入口检查源 Suite/worker 锁及 GPU 进程，不会停止现有任务。

源实验必须是当前 b07 续训目录，使用该目录届时最新断点；不使用历史840断点或诊断 Suite 作为训练源。两项已完成结果保留原字节及出处。源目录保持原有 b07 HEAD，目标使用独立工作目录。

在协作者现有代码目录执行以下准备步骤。`prepare_adaptive_checkout.sh` 只准备独立目录与共享本地运行环境，不启动 GPU 任务；它打印的通用 readiness 提示不是本候选入口。

```bash
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_CANDIDATE_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_CANDIDATE_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure
```

在选定 GPU 可用且源实验已正常暂停后，从刚准备的候选目录执行：

```bash
bash scripts/check_partitioned_attention.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --child-timeout 900 --stall-timeout 300
```

每个诊断子进程最多900秒；内部阶段300秒无变化会终止并回收该诊断自己的进程组。固定 micro32 遇到资源或数值问题会返回失败报告，不会自动减少样本或切回普通执行后宣布成功。

## 输出与判定

输出位于独立的 `runs/partitioned-attention-<时间>-<标识>/`。入口自动回传 `partitioned-attention.json` 到认证交换的 `qkan-mamba/partitioned-attention/<Suite>/`，验证 SHA-256 回执；上传失败保留原地报告与可重试 manifest。权重、输入、逐样本 trace、原始日志保持私有，不上传此类原始载荷。

报告含16/32恢复证据、源 checkpoint hash/step、设备/配置身份、完整更新数值/RNG、普通执行对照、Graph16/32原始AB/BA计时、准备/稳定显存快照、图回放与副本同步统计、精确回滚及子进程清理结果。只有两类速度对照和全部恢复/数值要求通过，顶层 `passed` 才为true。

诊断不会启动正式训练；通过后先审阅原生回传，再决定将分层路线接入续训。五个 baseline 的静态 fidelity 状态仍为 `not_verifiable`，既有正式准入要求保持生效。

当前实验回退就是从原 b07 目录用原入口续跑，它的代码、完整断点及已完成结果保持可用。实现层的逐文件补丁、候选副本、验证记录和可执行回滚另存于本机私有交付目录 `.runtime/partitioned-graph-candidate-20261114/`。

研究依据与边界计算见 [分层 Graph 研究](partitioned-retrieval-graph-research-20261114.md)。
