# 解析 K 梯度候选：独立恢复与真实轨迹预检

## 当前起点

候选 `bf16-key-bias-analytic-zero-v1` 已在原step14268、固定128条训练样本的单更新对照中通过原四项数值门限，BF16全局Adam相对L2为0.027103537600872692。该结果见[认证复核](../evidence/key-bias-native-review-20261116.json)。本轮沿用同一候选及[机制契约](attention-attribution-key-bias-recovery.json)，增加独立的[恢复/轨迹v2计划](attention-attribution-key-bias-recovery-plan.json)，保留历史筛查计划。

本候选显式改变梯度定义：每个反向回调仅把四层合并偏置的K段置零，Q/V保持原值；参数仍属于原AdamW组，历史矩、step、eps、lr及权重衰减保留。旧BF16的恢复结果不自动覆盖这个新策略。

## 本轮实际执行

一个父进程持有原supervisor/suite/worker排他锁，只在GPU0空闲时依次启动四个独立子进程。GPU已有工作时停止预检，保留原进程。全程使用已有离线环境、原缓存、已保存的账户私密交换配置，不安装依赖、不访问公共下载源。

| 子进程 | 计算 | 核对 |
|---|---|---|
| uninterrupted | 原dropout、micro16/logical128，连续两次完整更新 | 模型/梯度/Adam/样本调度/位置/排列/epoch统计/RNG完整哈希及分量哈希 |
| interrupt | 同一起点一次完整更新，保存私有状态并重开核对 | 显式新梯度策略、请求/代码绑定、原checkpoint和起点状态哈希 |
| resume | 新进程重建原运行时，再加载该诊断的中断状态并执行一步 | 与连续两步的完整状态、全部更新输出及样本身份精确相同 |
| trace | 固定原批次前两条样本；FP32原模型对照、带观察器的FP32、BF16解析候选及两条路线精确重放 | 真实sample-trace.v2结构、完整值、语义、原坐标读取与重放 |

恢复每个完整更新四层各8次回调，共32次；每次更新独立注册和移除hook。私有中断状态为 `qkan-key-bias-private-state.v1`，同时封装新策略、原checkpoint、原起点、请求和完整状态身份。只允许本次目录中显式指定的 `interrupt/diagnostic-state.pt`，不接收旧BF16或正式训练checkpoint作为此恢复输入。连续/恢复一致后，只删除该临时中断文件和匹配回执；失败时保留排查材料。

## 真实轨迹范围

样本选取规则在输出前冻结为原checkpoint的 `permutation[cursor:cursor+2]`，即已审阅128条训练样本的前两条，所有路线相同。训练标签仅在产生logits之后由交叉熵读取，无验证集或测试集访问，不选best-validation模型。

每条轨迹保存两篇文档的完整4000位置token、PAD mask、位置、第一层投影Q/K/V及进入SDPA的真实值、注意力输出、编码器输出、汇聚、logits、实际损失和投影梯度；还保存四层完整384维偏置的原始/修正/裁剪后梯度、参数与实际增量，以及Adam一二阶矩和step的前后值。BF16值精确转换为FP32后落盘，源dtype另记，不物化4000×4000注意力矩阵。

轨迹用两条样本的**联合平均交叉熵**反向和一次诊断更新，dropout=0；每层只触发一次leaf偏置回调。参数梯度/矩/增量属于这个固定两行批次，在每条样本记录中重复并明确标为共享批次量，不声称是逐样本参数梯度。这与恢复的logical128、原dropout是不同实验单元，结果不合并。

FP32观察器适配器必须先与原FP32 MHA实际前向、损失、全部53参数梯度和增量匹配（相对差异上限1e-5），才生成基线轨迹。各路线从相同完整起点精确重放，核对全部采集值和更新后完整状态；不要求FP32和BF16跨路线逐值相同。两路线还核对同样本、同输入和标签。所有`.npy`重开后逐值对照生产者值，并核对哈希、形状、dtype、全坐标覆盖和真实读取器首/中/末坐标；检查原数据行、mask和位置，使用实际裁剪梯度及原betas独立核对Adam矩递推和实际偏置增量。

Adam矩方程复核使用相对容差2e-5，一阶矩绝对容差1e-12、二阶矩1e-18，容许真实FP32运算舍入；实际偏置增量与更新前后值则逐值精确核对。这里的方程复核不等同于全部算子逐位重算。

本阶段强制efficient-SDPA后端，检查四层双文档调用和dtype；完整更新的profiler算子证明、短轨迹、匹配计时和收敛仍属后续阶段。当前架构状态保持 `approved`、trace readiness保持 `blocked`，直到实际回传并审阅真实fixture。Owner的替身文件不作为真实fixture或native通过证据。

## 合作者执行命令

原训练先按既有流程暂停，确认GPU0空闲，再在tmux运行。下列命令准备独立checkout并执行本预检，不自动启动正式队列：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_RECOVERY_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_RECOVERY_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_recovery \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --screen-from /root/projects/qkan-mamba-lra-adaptive-91b8df662c39/runs/key-bias-20261116T090926Z-60f6f2a2 \
  --optimizations experiments/baseline_delivery/optimization-presets/key-bias-consistency.json \
  --child-timeout 900 --stall-timeout 300
```

必须保留已完成筛查的 `key-bias.json`、`screen-input.json`，以及其引用的旧归因报告、请求和私有配对文件。原checkpoint、配置、数据、cursor和元数据仍须匹配已审阅14268；源训练若已推进，预检会停止，不切换样本或回滚正式训练来迁就本命令。各子进程900秒上限、300秒停滞上限，仅监督器自己创建的进程组会被清理。

## 输出、回传与回滚

新目录为 `runs/key-bias-recovery-<时间>-<ID>/`，自动把 `key-bias-recovery.json` 上传到认证交换的 `qkan-mamba/key-bias-recovery/<目录名>/` 并核对SHA256回执。紧凑报告包含四子进程退出和清理、精确恢复证明、两路线fixture路径/哈希、完整值核验与重放结果。原checkpoint、元数据和此前证据前后哈希必须保持不变。

私有完整值留在 `trace/trace-fp32/` 和 `trace/trace-bf16_analytic/`，每条路线上限128MiB；中断状态上限128MiB。原数据、权重、完整轨迹和私有张量不自动上传。所有诊断模型、梯度、优化器、样本时钟和RNG在finally恢复到原完整起点，hook和实例观察器移除，提交的训练样本/更新均为0。诊断状态永不用于正式训练恢复。

子进程失败也先保存部分报告、代码位置及退出信息，再尝试回传。上传失败时保留pending manifest，可仅重试上传，避免重复计算：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory /absolute/path/to/runs/key-bias-recovery-RUN_ID \
  --experiment key-bias-recovery --file key-bias-recovery.json
```

即便本轮通过，`candidate_admitted=false`、`formal_training_started=false`仍保持；后续短轨迹、匹配完整更新时间、收敛和最终接入契约另行验证。
