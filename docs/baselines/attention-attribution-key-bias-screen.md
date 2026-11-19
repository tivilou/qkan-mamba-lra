# K 偏置解析零梯度候选：首轮数值筛查

## 这次实际改变什么

候选ID为 `bf16-key-bias-analytic-zero-v1`，插件为 `gradient.key_bias_analytic_zero@1`，梯度策略为 `analytic_zero_k_bias_keep_adam_history_v1`。用户已启动本候选的设计和实现。本交付只提供一个受监督的数值筛查，不接入正式训练。

标准点积注意力中，`K_j=W_K*x_j+b_K`。对固定query，`q_i*b_K` 给全部可见key的分数增加相同常数；softmax抵消这个常数。因此，当掩码与参数无关、无额外bias_k/bias_v及其他改变位置关系的算子时，精确算术中的 `dLoss/db_K=0`。当前原模型满足本候选限定的结构。注意力dropout也不会改变这一精确恒等式，但本轮数值对照固定dropout=0。

有限精度BF16前向包含量化，实际程序不严格保持该不变性。本候选**明确采用新的梯度定义**：每个完整反向中，四层合并偏置的384维梯度复制一份，将 `[128:256]` 的K段置零；Q `[0:128]`、V `[256:384]`逐值精确保持。既有BF16前向、Q/K/V投影权重梯度及其他参数梯度保持原路线。这不是仅改变dtype，也不声称精确求出了BF16离散前向的导数。

保留全部53个参数张量、807170个元素。K偏置仍是原可训练参数，正常属于AdamW参数组，梯度为真实零张量而非None。原lr、eps、betas、weight decay、一二阶矩及step完整加载，不重置或冻结。零新梯度时，原一阶矩按beta1衰减、二阶矩按beta2衰减，step递增，原动量与权重衰减继续产生真实权重更新。所有参数的真实更新仍进入原全局比较。

## 对照与停止规则

固定同一原b07 checkpoint step14268、原permutation/cursor的128对训练样本、micro16、完整4000位置。一个子进程依次从同一完整状态执行：

| 路线 | 前向 | K偏置梯度 | 用途 |
|---|---|---|---|
| fp32 | 原MHA FP32 | 原程序 | 唯一全局准入参考 |
| fp32_analytic | 原MHA FP32 | 解析零策略 | 隔离梯度策略本身的影响 |
| bf16_analytic | 原BF16 SDPA | 解析零策略 | 新候选 |

每条只做一次完整更新，每条前后完整状态均检查哈希。新计算的原FP32路线必须与上轮私有配对文件的FP32前向、梯度、矩、实际增量精确匹配。两条解析路线的前向和非K梯度/矩/增量必须分别精确匹配已保存的原FP32和原BF16路线；后者不重跑未修改的BF16。原批次两路线裁剪均未改变梯度，本轮检查包括裁剪前后；若该一致性不成立就停止。

每层每条解析路线必须实际触发8次leaf hook，记录原始/修正后的完整384维回调值；每次检查Q/V相同、K为零，结束后移除本次注册的所有hook。已有其他参数hook时停止，防止组合语义不明。

分别报告FP32解析对照和BF16新候选相对原FP32的logits、损失、完整梯度和完整Adam增量差异。原门限保持2%/0.02/5%/20%，并重建全部53参数的全局指标。不会因K是理论零方向而把它从指标中排除。

这一步不产生新策略的跨进程恢复、短轨迹、sample-trace.v2或性能结论。即使两组数值门限通过，仍需后续独立验证恢复/完整轨迹/匹配完整更新时间及收敛。计划状态保持approved，trace readiness为blocked，正式准入始终关闭。旧诊断的恢复结果不继承给新梯度策略。

## 执行

在原训练按既有流程暂停、GPU0空闲后，于师弟tmux中执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_KEY_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_KEY_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_diagnostic \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --attribution-from /root/projects/qkan-mamba-lra-adaptive-ebe7e4732dc0/runs/attention-attribution-20261116T024408Z-e40dd283 \
  --optimizations experiments/baseline_delivery/optimization-presets/key-bias-consistency.json \
  --child-timeout 900 --stall-timeout 300
```

复用离线环境、原缓存和已保存的私密交换配置，不安装依赖。不自动停止其他GPU任务；先检查空闲GPU，再以原supervisor/suite/worker锁排除活跃训练。原source、checkpoint、配置、样本位置及数据身份须与已审阅归因完全一致；原训练若已推进，停止而不切换测量位置。子进程严格900秒总时限、300秒无阶段推进时限，仅清理自己创建的进程组。

## 输出、回滚与隐私

自动回传新目录的 `key-bias.json` 至 `qkan-mamba/key-bias/<目录名>/`，核对SHA256回执。`screen_complete=true` 表示三路线测量与回滚完成，`numerical_passed` 分别表示两个对照结果；**完成不等于数值通过**，`candidate_admitted=false`、`native_trace_ready=false`始终保留。

完整三路线参数证据及全部64次回调的原始/修正值留在 `screen/paired-updates.pt`，上限128MiB，保存后重开核对完整值哈希。文件为 `{routes: ..., callbacks: ...}`，包含梯度、矩和真实增量；它不是训练checkpoint，不上传。原数据、checkpoint及此前配对文件只读，不复制到新训练源。实例hook在异常路径也移除，完整模型/梯度/Adam/样本时钟/样本位置/RNG状态在finally中恢复并验证哈希。

上传失败保留本地JSON与pending manifest，只需重试上传：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory /absolute/path/to/new/runs/key-bias-RUN_ID \
  --experiment key-bias --file key-bias.json
```

下一阶段如需保存新策略的可恢复训练状态，必须显式绑定本梯度策略ID和版本；本次不会将原恢复契约静默当作新策略的认证。

机制记录：`attention-attribution-key-bias-recovery.json`。独立v2计划：`attention-attribution-key-bias-plan.json`。Owner只用QT2标准库/替身验证控制流程、标量恒等式和回滚。

## 已完成的真实数值筛查（2026-11-16）

上述筛查已由合作者执行，无需重复。执行修订91b8df662c39，Suite为`key-bias-20261116T090926Z-60f6f2a2`。认证报告32412字节，SHA256为`1bcffacdb03a8e2ea501135422ad543dbb6aaffdb5d91868ceb1179b26108edc`。FP32解析对照/新BF16候选全局Adam相对L2分别为0.0016498362797680208和0.027103537600872692，四项固定门限均通过。全部53参数保留、原历史矩/步数一致、非K及前向匹配旧路线、64次回调和精确回滚通过；原训练checkpoint保持不变。

详情见[审阅证据](../evidence/key-bias-native-review-20261116.json)及[数值检查点日志](../research-log/20261116T-090926Z-key-bias-numerical-screen.md)。本阶段仍没有新策略恢复、sample-trace.v2、短轨迹、匹配计时或收敛结论；正式准入和trace readiness继续关闭。
