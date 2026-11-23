# Mamba BF16 数值与恢复短验证

## 当前下一步：仅 FP32 重复性对照

已有回传表明两条 BF16 路线在中断前第2步就出现独立运行差异，中点加载本身完全一致。使用同一入口的 `--fp32-repeat`，只跑两个全新 FP32 进程、每个2步，总共4次一次性更新，保持原micro/logical/sample-clock/dropout。每步返回完整状态组件hash和损失，能发现“损失相同但梯度/参数已不同”。不启动BF16或Graph，也不重跑旧测速/40步验证。

在服务器执行整段（使用绝对路径，任意当前工作目录均可）：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-ff58cd8feab6 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_REPEAT_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_REPEAT_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_REPEAT_REV:0:12}" &&
cd "$MAMBA_REPEAT_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --fp32-repeat \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "$QKAN_CACHE" --gpu 0
```

检查点SHA必须等于已审计BF16报告的输入（step24981），源若已推进则停止，不回滚源。两个进程初始完整状态及每步样本/损失/组件都精确一致才返回0；观察到差异返回76，同时显示首次差异步并自动回传 `qkan-mamba/mamba-repeat/<运行名>/mamba-repeat.json`。控制完成与重复性通过有区别，精确一致也不解锁BF16。新报告含旧审计hash以复用BF16证据；每子进程900秒/单阶段300秒，独占空闲GPU、源锁、退出清理沿用原实现。临时复制的检查点自动清理，原正式Suite和七个完成结果保留。

已审计测速见 [实测记录](../evidence/mamba-speed-native-review-20261123.json)：BF16 eager 吞吐 +45.22%，BF16 Graph +48.91%，两者单更新量相对差异约22.86%。本入口只补充数值和恢复证据，不重复四路线计时，不将Transformer阈值或K偏置策略用于Mamba。

## 执行指令

原训练目前保持暂停，无需再次 Ctrl-C。若已经恢复，则在原一体看板请求安全暂停并等入口退出。执行整段：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a33420b5408d &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_CHECK_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_CHECK_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_CHECK_REV:0:12}" &&
cd "$MAMBA_CHECK_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "$QKAN_CACHE" --gpu 0
```

新 worktree 复用现有环境与私密配置。原正式代码、检查点、七项已完成结果和其他模型均不变。只复制一个最新暂停的 Mamba 检查点；旧测速报告复用已提交的审计记录，不要求重传或重跑旧诊断。

## 固定预算和证据

- 固定接下来的8个完整训练更新，保留原物理micro和逻辑batch、不升档，保留dropout/AdamW分组/动量/样本学习率时钟/采样顺序。若当前epoch剩余行不足8步，明确停止，不换样本或偷偷跨epoch。
- 七个独立子进程：FP32参考8步；BF16 eager连续8步、4步中断、新进程恢复后4步；BF16 Graph同样8步与4＋4步。合计40次一次性更新，正式提交0步。每个子进程900秒上限、单阶段300秒上限，出错停止，不自动重复。
- 对每种BF16路线，记录相同8个训练批次的损失和行身份hash；第一步记录全部参数的原始累积梯度（裁剪前）及AdamW更新量，返回逐参数范数、差异、相对L2、余弦和最大绝对差。缺失梯度单独标记。原始数组不回传。
- 恢复检查比较连续8步与4＋4步的完整模型、optimizer、scheduler、RNG、梯度、permutation和epoch/cursor/step的hash、每步行身份及损失，并核对新进程起点等于中点。严格相等失败会显示失败，不降低规则。
- Graph捕获先于每个新进程的状态恢复；新进程恢复后继续原参数和随机状态。原训练的完整恢复还需实际worker接入验证，本次不是正式准入。
- 所有模型计算只访问train；val只建立原checkpoint契约所需的文件身份，未执行val/test前向。没有新的sample-trace.v2，不用此报告替代正式trace证据。
- 快照输入使用独立索引副本，避免推进8批次时改写原permutation视图。

回传 `qkan-mamba/mamba-validation/<运行名>/mamba-validation.json`，包括逐参数数值摘要、7个子进程退出/清理记录、连续恢复比对和源文件未改动结果。临时参考数组、中点状态和本次复制的checkpoint在finally清理，原源Suite权重始终保留。精简receipt和报告留存；原始异常文本与worker日志仅本地保存。

完成提示只表示固定前缀和恢复检查结束，不表示BF16已通过收敛或正式准入。对逐参数差异和8步损失进行数值审阅后，才决定实际worker接入；Image/Pathfinder输入形状另作短验证，MEGA随后，S5/S4D分别判断。

仅重传，不重跑：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/实际mamba-validation目录 \
  --experiment mamba-validation --file mamba-validation.json
```
