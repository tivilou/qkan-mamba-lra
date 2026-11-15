# 独立 BF16 注意力诊断

2026-11-15。PAD 固定前缀统计为零收益后，下一项最小实验是 BF16 efficient SDPA。代码已交付，真实 GPU 数据待返回。这是机制筛查，不是完整训练加速或收敛认证。所有选项默认关闭；只在 `attention_probe` 执行域允许 `bf16_amp`，旧训练入口和 PAD 统计入口仍拒绝启用。

## 测什么

使用原 b07 未完成 Retrieval 最新完整 checkpoint 的模型权重，以及 train 前 16 行的第一篇文档。保留宽度 4000、原位置编码、PAD=1 的键掩码、第一层 pre-norm 和 QKV 投影。只对一层、一篇文档的注意力前向和反向执行；上游梯度是固定独立高斯向量，不是完整模型的 loss 梯度。未使用标签值，未读取 val/test，未更新优化器，未提交训练样本。

比较 FP32 QKV 与显式转成 BF16 的 QKV，输出分别为 FP32/BF16，QKV 叶子梯度均回到 FP32。参数仍为原 checkpoint FP32。强制现有 `EFFICIENT_ATTENTION` 后端；未加入 Flash、外部包或网络下载。PyTorch 2.7.1 的 [efficient SDPA 检查](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) 包含 BF16 路径和掩码限制；实际可运行性以本次内核报告为准。

1. dropout=0 时比较 output/dQ/dK/dV 的相对 L2 与最大绝对误差。output 相对 L2 不超过 0.02、各梯度不超过 0.05 且全部有限，才进入计时。这些是预先固定的机制筛查阈值，不是训练数值协议。
2. 使用原训练 dropout，分别恢复 RNG 重跑每条路线，要求同路线输出、梯度及结束 RNG 精确一致。不要求跨精度位级相同。
3. 每条路线预热两次，然后两个 AB/BA 配对、每条路线四次测量。同步墙钟包括 QKV 转换、前向、反向和上游梯度转换；RNG 恢复、数值检查和 profiler 不计入该局部计时。
4. profiler 单独证明 efficient 前向/反向均出现，禁止静默改用 math/Flash；记录 allocated/reserved 峰值及相对存活输入的增量峰值。两条路线预热后共享 allocator，reserved 不代表独立路线的最小显存需求。
5. 结束恢复探针 QKV/RNG，检查叶子 `.grad` 未改变，重验源 checkpoint 与元数据 SHA。整个过程不写源 checkpoint。

局部吞吐收益只适用于这一层/文档的注意力。它不包括四层编码器、两文档、投影/FFN、优化器、数据传输、保存与 Graph，不能直接换算整任务 samples/s 或 ETA。通过且收益明确后，才设计单独的完整更新 BF16 候选，绑定 v2 机制 dossier、数值/收敛协议、独立进程恢复、sample-trace.v2 及 Graph16/32 同资源比较。

## 直接执行

在原训练已按既有流程暂停、GPU 0 空闲后运行。入口会核验 GPU 进程，非阻塞获取原 supervisor/suite/worker 锁；任一仍占用就退出，保留原任务。不要终止不明 GPU PID。复用原训练缓存与账号私密配置，无需新安装。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_PRECISION_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_PRECISION_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/profile_attention_precision.sh \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --optimizations experiments/baseline_delivery/optimization-presets/bf16-attention-probe.json \
  --micro-batch 16 --child-timeout 900 --stall-timeout 300
```

通用 checkout 脚本打印的 readiness 示例不属于本实验；继续执行上面的 `profile_attention_precision.sh`。这组命令只运行一次机制诊断，不启动 25 任务训练。默认先 micro16；本阶段不自动扫描 micro32 或重复调阈值。

输出 `runs/attention-precision-<时间>-<随机后缀>/attention-precision.json`，自动回传到 `qkan-mamba/attention-precision/<目录名>` 并校验 SHA-256 回执。失败也回传紧凑报告和代码位置，不回传异常正文、原始日志或 tensor。上传失败保留报告及重试声明：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/ATTENTION_PRECISION_DIRECTORY \
  --experiment attention-precision --file attention-precision.json
```

数值筛查未通过返回状态 76，运行/确定性/内核/超时错误返回非零；成功表示局部机制测量完成，收益可能仍为负。检查 `timing.throughput_gain_fraction`、`dropout_off_numerics`、`same_route_replay`、`kernel_proof`、`rollback` 和 `original_checkpoint_and_metadata_unchanged`。报告不将 BF16 纳入正式准入。

## 存储与验证边界

不复制 checkpoint、不生成新数据缓存、不导出 raw profiler trace 或 QKV。临时 GPU 张量随诊断子进程释放；本地仅留下请求元数据、聚合报告和有界诊断状态。复用 Split 会读取 train NPZ 与缓存哈希进行校验，可能存在一次磁盘读取成本，未计入注意力局部计时。

架构仍为 approved，真实 sample-trace.v2 全值/lineage/语义重放待完整候选实现。owner 仅运行标准库、替身与 CLI/shell 控制测试；不导入 Torch/NumPy，不运行模型/GPU。可关闭 BF16 直接回到原训练版本，原始两个已完成结果及最新 b07 完整 checkpoint 均保留。
