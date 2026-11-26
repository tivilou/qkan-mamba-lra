# MEGA FP32 FFT 补齐短测

先测一个具体候选：ListOps 的双向 EMA 使用长度 8190 的 FFT，候选补零到 8192，再保留原卷积截取。模型仍为官方 MEGA，完整注意力、FP32、micro-batch、逻辑 batch、Adam 和样本预算均保持一致。补齐在数学上保留该线性卷积，但浮点误差和实际吞吐必须分别测量；当前没有原生测速结果。

插件仅挂在临时模型的 EMA 实例上，并校验固定官方源码哈希。训练核心、共享上游源码、已有 Suite 和正式策略都不修改。子进程使用已有 `.venv-mega`（Torch 1.13），不安装或下载依赖，不套用 Transformer 的 Torch 2 编译路径。

## 实验机执行

先在原训练终端按一次 **Ctrl-C**，等检查点保存、入口退出，再粘贴整段。不要与正式训练同时占用 GPU 0。已暂停则直接执行。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
MEGA_DIAG_REV=$(git rev-parse FETCH_HEAD) &&
MEGA_DIAG_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MEGA_DIAG_REV:0:12}" &&
cd "$MEGA_DIAG_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mega-fft -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-89353fc83f89/runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0 &&
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last
```

原 Suite 当前须有未完成的 `mega-listops-s42` 完整断点。若任务已结束，不重复它；选择一个已有断点的未完成 MEGA 任务，在诊断参数中添加相应 `--job mega-text-s42` 等。没有断点的任务不在本次短测范围。

这是一个入口串联短测和原训练：短测打印阶段心跳，正常完成并验证回传后恢复原训练、美化看板及 Retrieval 最后顺序。无论测得正收益还是负收益，均恢复原策略。诊断失败或回传失败会停在原暂停状态，保留紧凑报告供排查；再次续跑使用[原统一入口](experiment.md)。

## 结果与边界

- 一个原生子进程，最多 900 秒、单阶段无进展最多 300 秒；这些是超时上限，不是预计时长。
- 取最新完整断点的一个待训练逻辑 batch。每次恢复独立克隆的模型、Adam、RNG 和样本调度状态；独立检查 Adam step 起点及增加量，避免可变引用污染对照。
- 3 次数值观测兼预热，随后 ABBA×2 的 8 次计时，共 **11 次临时更新、0 次正式更新**。数值比较包括参数、更新量、裁剪后梯度、Adam 状态及 RNG／调度一致性，并提供原策略重复误差作参照。
- 计时涵盖 CPU 取数、H2D、前后向、裁剪、Adam、样本调度和同步；不计恢复和数值副本开销。本次不测数据预取重叠，不承诺整轮加速或收敛等价。
- 记录峰值显存、实际 FFT 调用和长度。结束时精确恢复临时状态，移除自有检查点副本，并验证原文件哈希保持一致。
- `runs/mega-fft-*/mega-fft.json` 自动回传到 `qkan-mamba/mega-fft/<运行名>`，沿用现有 SHA-256 回执工具。回传失败可重传同一报告，无须重做测速。

收到结果后先看完整更新吞吐，再看候选误差与原策略重复噪声。短测不打开正式策略；如有明确收益，再决定是否做实际保存／恢复验证。无需重跑其他模型的全量检查。
