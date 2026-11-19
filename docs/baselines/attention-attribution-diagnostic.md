# Adam 更新差异专项诊断

上次 `c6398e3` 的同 checkpoint 检查已通过跨进程恢复，BF16 的 Adam 更新相对差异为 21.0526%，超过固定 20% 门限。本入口只定位这项差异；原门限和 BF16 默认关闭策略保持原样。

固定读取上次拒绝报告、原请求及原 b07 checkpoint。先按报告 SHA256 校验，再核对 checkpoint、数据、cache、配置和元数据完全一致；原训练若已推进，诊断会停止，不自动切换到新训练位置。GPU 必须空闲，原 supervisor/suite/worker 锁必须可获得。入口只检查并保留其他进程。

只启动一个受监督子进程，FP32 与 BF16 各做一次 dropout=0 的完整更新：micro16、logical128、原 permutation/cursor 的同128个训练样本、完整4000位置。复用原 Classifier、交叉熵、裁剪、AdamW 和 sample clock。通过临时实例 hook 观察真实 Adam step，不改其梯度或计算；结束和异常时恢复 hook，并恢复完整训练状态，核对精确哈希。

采集全部参数的裁剪前后梯度、一二阶矩前后值、step计数、更新增量、实际 step 时使用的 lr/eps/betas/weight decay。报告给出绝对 L2、相对 L2、分母是否被下限截断、最大绝对差、cosine、符号差异、零值数、每个参数和层对全局误差平方和的贡献，并重建原聚合梯度/更新指标。缺失的初始 Adam 状态会明确列出，不伪造为观测值。参数排序属于事后描述，不能当作因果消融。

## 执行

在师弟服务器 tmux 中，原训练按既有流程暂停且 GPU0 空闲后复制执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_ATTR_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_ATTR_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_attention_update.sh \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --optimizations experiments/baseline_delivery/optimization-presets/bf16-attention-update.json \
  --attribution-from /root/projects/qkan-mamba-lra-adaptive-c6398e34b89c/runs/attention-update-20261115T132955Z-1ba0ff60 \
  --child-timeout 900 --stall-timeout 300
```

复用已存在的离线依赖与私密交换配置。只自动回传 `attention-attribution.json` 到 `qkan-mamba/attention-attribution/<目录名>/`，并校验 SHA256 回执。失败同样回传紧凑报告。上传失败保留 pending manifest，可用原 `return_artifacts` 命令重试上传这个 JSON，无需重新运行计算。

完整配对参数值保存在本次目录的 `attribution/paired-parameters.pt`，上限128MiB，生成后重新打开验证全部值哈希；它是私有诊断证据，不自动上传，不作为训练 checkpoint。原数据缓存和 checkpoint 不复制。完整值是参数级证据，不代替 `sample-trace.v2` 的样本级认证。

成功只表示参数级诊断完成：顶层 `attribution_complete=true`，`candidate_admitted=false`，`passed=false`（原候选不被重新准入）；子结果的 `numerical_passed` 独立记录固定门限结果。不会重复恢复检查、短轨迹、trace、profiler 或测速，不启动正式训练或 BF16+Graph。

契约：`docs/baselines/attention-attribution-contract.json`。Owner 仅用 QT2 标准库/替身和真实 CLI 验证控制流程，真实参数归因需本次回传后审阅。
