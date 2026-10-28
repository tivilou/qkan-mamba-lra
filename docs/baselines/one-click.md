# 协作者一键运行入口

入口：`bash scripts/run_baselines.sh`。Linux + Python 3.10/3.11；GPU 阶段需要 NVIDIA 驱动，Mamba1 还需要与 PyTorch cu126 对应的 CUDA 工具链和 `nvcc`。首次安装和作者数据下载需要网络。脚本创建项目内 `.venv-baselines`，固定 PyTorch 2.7.1、NumPy 2.2.6、einops 0.8.1；不改系统 Python。

**当前交付状态：研究方 CPU 工程验证已通过，合作者 GPU 预检待执行，尚未获准正式比较。** 2026-10-29 在 3080Ti 调度节点通过统一队列完成了单测、Transformer/S4D minimal 新进程精确恢复、trace 语义与重放，以及 `mega-document-disjoint-v3` 的 15 个 NPZ split、3 份字典身份和 15 份配置构建检查。可审计收据见 [`docs/evidence/baseline-runner-diagnostics-20261029.json`](../evidence/baseline-runner-diagnostics-20261029.json)。这些检查全部标记 `formal=false`，不是 LRA 性能结果，也不能替代实际 GPU 预检。

`release.json` 仍明确记录未冻结 cohort、未认证的模型；数据资格也仍是 `insufficient`。现在可以准备数据、执行 GPU 诊断，`--mode run` 会说明原因并拒绝正式训练。此文档不是“baseline 已复现成功”的证明。

## 1. 获取代码和私有配置

从 GitHub 的 `research/collaboration-foundation` 分支取得双方确认的干净 commit；运行 `git pull --ff-only origin research/collaboration-foundation`，再记录 `git rev-parse HEAD`。同一 suite 恢复必须使用相同 commit 和协议。

通过私下提供的渠道取得交换服务地址、token 和 CA 文件，在 shell 中设置：

```bash
export PROJECT_EXCHANGE_URL='https://<私下提供的主机>:18084'
read -rsp 'Exchange token: ' PROJECT_EXCHANGE_TOKEN; echo
export PROJECT_EXCHANGE_TOKEN
export PROJECT_EXCHANGE_CA_FILE='/absolute/path/exchange-ca.crt'
```

不要将凭据放到 Git、截图、命令行参数或实验配置中。首次 CA 文件通过可信渠道提供；脚本保留 TLS 主机名和证书校验。

## 2. 自动准备

```bash
bash scripts/run_baselines.sh --mode prepare --cache /data/qkan-cache
```

脚本自动从项目交换服务下载并校验上游代码包，再从作者地址 `https://dl.fbaipublicfiles.com/mega/data/lra.zip` 下载固定数据归档（约 1.45 GB），依次重建 v1 → pairclean-v2 → document-disjoint-v3，最后逐个校验 15 个 NPZ。已有正确文件复用；作者下载支持断点，交换对象在中断后重下该对象。构建阶段原子提交，中断的阶段保留并更名后重新构建。预留至少 12 GB 磁盘，失败重试时需额外空间存放保留目录。

已有同一 v3 数据可加 `--data /absolute/path/mega-document-disjoint-v3`；仍会验证每个分割。Retrieval 使用新的文档组独立划分，因此不可直接与官方或历史 LRA 分数对比。

原 LRA 对 AAN 文本的转发有限制，**我们的服务不发布该文本或整个数据压缩包**。各任务的用途证据和正式比较资格详见 `docs/data-use-boundaries.md`。作者下载失败时保留缓存；脚本不会自动切换为未审查的第三方数据集。

## 3. 合作者实际 GPU 预检

```bash
bash scripts/run_baselines.sh --mode preflight --cache /data/qkan-cache \
  --gpus 0 --models transformer,s4d_minimal,mamba1 \
  --tasks listops,text,retrieval,image,pathfinder --seeds 42
```

这会在指定的每块 GPU 上，对每个任务/模型/种子执行有界的真数据诊断：仅训练集前三个逻辑 batch、验证集前两行、两轮，不读取测试集。比较不中断与“新进程中断后再用另一个新进程恢复”的完整状态，并验证所选验证样本的 `sample-trace.v2` 结构、完整数值和分类头重放。诊断分数不进入正式结果页。研究方 CPU 诊断已通过，但失败仍必须保留日志排查；尤其 Mamba CUDA 算子及确定性支持尚不能由 CPU 测试代替。

`s4d_minimal` 是作者明确标为教学用途的 minimal S4D 模块加本项目分类适配层，绝不等同于完整 S4/S4D benchmark 复现。当前三个实现是工程候选；MEGA、S5 等未交付，正式 cohort 需另行冻结。

## 4. 正式放行后的一键实验

研究方补齐数据资格、固定 cohort/协议并提交逐模型复现证书后，更新 `release.json`。合作者更新到该发布 commit，再执行：

```bash
bash scripts/run_baselines.sh --mode run --cache /data/qkan-cache \
  --gpus 0,1 --models transformer,s4d_minimal,mamba1 \
  --tasks listops,text,retrieval,image,pathfinder --seeds 42,43,44
```

每块 GPU 一次一个独立 job；seed/任务/模型独立目录。配置固定训练预算、有效 batch 和 microbatch，microbatch 梯度累积；不因 OOM 静默改参数。按验证准确率选择最优权重，测试阶段重新载入该权重。Token ID 使用 embedding 和 padding mask，Retrieval 两文档独立编码、共享权重后进入显式 pair head。

入口打印 `Suite: /.../runs/baseline-...`；保存路径。单任务失败会停止启动新任务，已在运行的任务完成后整体以失败状态返回，已完成任务保留。

恢复或补传：

```bash
# 参数选择必须与原 suite 一致；Ctrl-C/SIGTERM 会请求在更新边界保存。
bash scripts/run_baselines.sh --mode run --cache /data/qkan-cache \
  --gpus 0,1 --models transformer,s4d_minimal,mamba1 \
  --tasks listops,text,retrieval,image,pathfinder --seeds 42,43,44 \
  --resume /absolute/path/to/runs/baseline-SUITE

# 计算已完成，但网络或服务上传失败：只重试上传，不重训。
bash scripts/run_baselines.sh --mode upload --resume /absolute/path/to/runs/baseline-SUITE
```

最新 checkpoint 含模型、优化器、调度器、全部 RNG、epoch/下个样本游标、数据排列、验证选择和最优权重；原子指针保留前一代。崩溃时从最近已提交更新恢复，最多重算自上次 checkpoint 之后的更新。损坏文件不会自动跳过；保留旧代供显式恢复。

## 5. 结果回传和分析

计算完成后自动将配置、指标、最优权重、最新完整状态和 trace 校验摘要上传至 `qkan-mamba/baseline-suite/<suite-id>/`。超过 512 MiB 的文件分块，附原文件和各块的 SHA-256 清单；客户端按块上传、核对服务收据，失败可补传。

`compute-complete.json` 和 `delivery-complete.json` 区分“已算完”与“已交付”。原始输入、完整 sample trace、原始日志不会自动发布；保留在合作者服务器，按后续具体 case study 的权限和需求单独交换。小型指标经研究方审查、合并多种子统计后同步 GitHub 与门户；诊断测试不当作正式性能。
