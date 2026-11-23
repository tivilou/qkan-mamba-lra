# Mamba1 加速短诊断

四路线测速已完成并审计，下一步使用 [BF16数值与恢复短验证](mamba-validation.md) 的单条诊断指令；无需重复本页测速。

当前顺序：先 Mamba1 Retrieval，再将被证实有效且兼容的实现用于 Image、Pathfinder；然后处理 MEGA。S5/JAX 和 S4D 分开判断。已完成的七项保持原结果，不重跑。

本次交付是测量入口，尚无新的 GPU 提速结果，也不自动放行新的正式策略。不要给当前训练目录直接加 BF16 或改写绑定配置。

## 在执行服务器运行

1. 在当前训练看板按一次 Ctrl-C，等待入口完整退出，确认保存和清理完成。不要直接杀 worker。
2. 在原目录执行以下整段。新代码放独立 worktree，沿用现有环境和私密配置；只复制所选 Mamba 最新完整检查点，不复制其他任务权重。若任务已经完成，入口停止，不回退历史进度。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-14d681111173 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_DIAG_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_DIAG_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_DIAG_REV:0:12}" &&
MAMBA_SOURCE_SUITE="$PWD/runs/bf16-graph-formal-20261119T014456Z-1cb7160c" &&
cd "$MAMBA_DIAG_ROOT" &&
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-speed -- \
  --source-suite "$MAMBA_SOURCE_SUITE" --cache "$QKAN_CACHE" --gpu 0
```

这是短诊断，不启动正式训练；终端显示四条路线和子进程阶段。每个子进程上限 900 秒，阶段停滞上限 300 秒，总预算最多四个子进程；基准失败即停止。任何失败都保留原训练状态，原入口保持暂停。

四条路线为 FP32 eager 对照、BF16 eager、FP32 Graph、BF16 Graph。每条均从同一检查点、同一训练行、同一物理 micro/logical batch 出发，先预热再进行两轮 ABBA 共八次完整更新。每次恢复模型、原 AdamW 分组/动量、样本学习率时钟与 RNG，不推进源检查点或测试集。

计时包括 CPU 取数、固定内存传输、前向、反向、梯度裁剪、AdamW、学习率时钟和 GPU 同步；不含捕获、恢复、检查与序列化。该短路径未复现生产预取重叠，因此只是候选筛选，不能直接推算整轮训练速度。记录捕获成本以判断收益回本。Graph 使用完整分类器前后向捕获，原 optimizer 在捕获外；没有复用 Transformer 专用 K 偏置策略。

回传 `qkan-mamba/mamba-speed/<运行名>/mamba-speed.json`：四组完整步耗时、模型及更新量相对误差、损失、RNG/状态相等情况、实际 Graph 回放次数、原 micro/logical batch、捕获成本与清理记录。`use_fast_path` 仅是模型标志，不当作实际融合内核执行证明。原始日志和权重不回传。

## 接入顺序

- 先看 Retrieval 报告。没有收益就保留原路线；正收益候选再做针对性的数值/实际 worker 恢复验证与生产路径复测。3% 且至少三对正收益只是筛选信号，不是科学准入门槛。
- BF16 改变算术，不凭损失接近或权重总体误差小就准入；必须关注更新量误差、梯度、短前缀和原检查点兼容性。
- Image/Pathfinder 输入形状不同，需要各自短测。入口可显式选择 `--job mamba1-image-s42` 或 `mamba1-pathfinder-s42`，当前版本要求已有完整检查点，不伪造未开始任务的状态。
- 正式接入保持一条续跑命令和原看板，保留七个完成结果及其他模型原执行器；本次不提前创建新的正式 Suite。
- MEGA 后续使用自己的算子/数值路径，不套用 Mamba 测量结果。S5 保留原 JAX JIT 语义，S4D 单独检查 FFT/复数运算。

只重传报告而不重跑诊断：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/实际的mamba-speed目录 --experiment mamba-speed --file mamba-speed.json
```

若不切换加速，回到原目录使用原 `experiment.py resume --suite ...` 即可从其最新检查点恢复，继续显示原看板。

## 首次基准失败与诊断返回

`mamba-speed-20261123T013728Z-d34843dd` 的 FP32 基准在约12.8秒后以1退出，没有测速数据。源检查点不变，子进程组已清理。旧版末尾的 returned 只表示报告上传，不表示测试通过。修订版明确显示 FAILED/incomplete，回传异常类型与项目代码位置，并保持退出码76。原始异常文本、日志和张量仍留在本地。

新诊断补齐与原正式 worker 相同的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。这是已确认的启动环境遗漏；首次实际异常原因仍需读取旧 `fp32_eager.log`，不应从心跳堆栈推测。修复不更改原训练代码、七个完成结果或任何检查点。
