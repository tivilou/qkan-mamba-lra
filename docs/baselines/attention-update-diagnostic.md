# 可选 BF16 注意力完整更新诊断

这版把已实测有效的局部 BF16 注意力扩展到四层、双文档的真实训练更新。它是独立诊断候选，默认关闭；原训练入口与两个已完成结果保留。上一轮第一层、单文档、Gaussian cotangent 的约 2.94 倍收益仅是机制证据。本版整步收益、数值和恢复状态要以新的服务器报告为准。

`bf16_amp` 是沿用的配置键，本版实际只将 SDPA 的 Q/K/V 转为 BF16。原 FP32 QKV 投影、输出投影、残差、FFN、LayerNorm、pair head、交叉熵、master 参数和 Adam 状态继续用 FP32。每个模型实例临时替换自己的 `_sa_block`，退出恢复属性；没有全局方法替换或新参数命名。`false` 的空策略直接保留原实例。现有 formal worker 不接入此候选。

配置入口为 `optimization-presets/bf16-attention-update.json`，只允许 `attention_update` 诊断作用域。PAD、Flash、Graph 组合会被拒绝。关闭候选即使用原 b07 训练入口；这不需要修改原 checkpoint 或迁移已完成任务。

## 固定验证协议

- 固定 micro16、logical128、完整 4000 位置，不调 batch、不改变样本顺序、Adam 更新频率或按样本调度学习率。真实更新从原 checkpoint 的 permutation/cursor 开始。至少需要距 epoch 尾部还有 512 行；不足时报告约束，不跨越原验证边界。
- 先运行 BF16 连续两步、一步中断、全新进程恢复一步，比较 model、gradients、Adam、样本调度、permutation/cursor、epoch 统计、Python/NumPy/Torch/CUDA RNG 与精度策略的完整值哈希。所有子进程结束前回滚到各自原始状态。诊断 checkpoint 使用独立 schema，正式迁移入口不接收。
- dropout=0：先核对 FP32 实例适配器与原 MHA 的完整更新误差各项不超过 `1e-5`；再比较 BF16 的真实 logits、CE、未裁剪全模型梯度和 Adam 参数增量。预先固定门限依次为相对 L2 2%、CE 绝对差 0.02、梯度相对 L2 5%、Adam 增量相对 L2 20%。失败停止，不自动放宽门限。
- 以相同连续训练行做四步 dropout=0 的短轨迹筛查，每步 CE 绝对差不超过 0.10。它测试精度误差的短期累积，**不认证长期收敛**。原 dropout 用于恢复与速度测量，不要求 FP32/BF16 之间 RNG 或轨迹逐位相等。
- 固定 train 行 0、1，分别生成 FP32/BF16 私有 `sample-trace.v2`：完整 tokens/mask/positions、第一层每个文档的投影 QKV、实际 SDPA QKV、attention、真实损失 QKV 梯度、encoder/pool/logits/loss，以及批次共享的投影参数梯度。共享参数梯度来自两行的 mean CE，不伪装成独立单样本参数梯度。BF16 原值精确转存为 FP32 数组并标注来源 dtype。每条路线重新从原状态实际前向/反向一次，核对所有完整值，再重开文件做共享结构验证、哈希/shape/dtype、原坐标、来源行与任意坐标读取检查。这里的 checkpoint 是 original latest，绝不标作 best-validation。
- 原 dropout：两路线各预热两步；固定两组 AB/BA，每路线四次同步完整更新，比较中位数。计时包含 CPU minibatch gather、H2D、真实 forward/loss/backward、finite/clip、Adam、样本 scheduler；不含加载、状态恢复、预热、trace 和 profiler。FP32 和 BF16 都强制既有 efficient SDPA，另在计时外检查 ATen 前后向证据与全层 BF16 调用数。关闭 MHA fastpath 供两路线匹配，子进程退出时恢复。没有 Inductor、Flash 或 CUDA kernel-name 导出声明。

速度对照是同 micro16 的 eager 完整更新，**不是正在使用的 Graph16/Graph32**。任何正确性/trace/恢复失败都不准入；速度非正收益返回 76。通过只表示这个诊断候选通过，尚不启动 25 任务训练，也不证明 BF16+Graph 或长期收敛。

## 直接执行

在原训练按已有流程暂停且 GPU 0 空闲时，在师弟服务器的 tmux 中复制执行。入口只检查占用，不终止现有进程；非阻塞获取原 supervisor/suite/worker 锁。复用已有离线环境、数据缓存和账号私密交换配置，无新增安装。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_UPDATE_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_UPDATE_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_attention_update.sh \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --optimizations experiments/baseline_delivery/optimization-presets/bf16-attention-update.json \
  --child-timeout 900 --stall-timeout 300
```

通用 checkout 脚本打印的 readiness 示例不是本次入口；继续执行上述 `check_attention_update.sh`。每个子进程上限 900 秒，阶段不推进 300 秒触发诊断；阶段变化会更新心跳。监督器只清理自己启动的进程组。正常子进程也核实退出、回收及空进程组，不留下旧 CUDA 占用。

紧凑报告自动回传到 `qkan-mamba/attention-update/<目录名>/attention-update.json`，校验字节数及 SHA-256 回执。重点看 `recovery.passed`、`results.benchmark.numerical`、`trajectory`、`trace`、`timing`、`rollback`、`child_executions`、`original_checkpoint_and_metadata_unchanged`。失败同样回传报告与有限代码位置；上传失败保留 pending manifest，可重试：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/ATTENTION_UPDATE_DIRECTORY \
  --experiment attention-update --file attention-update.json
```

## 存储与交付边界

读取原最新 checkpoint，始终不复制或改写它；不复制数据缓存，不导出原始 profiler trace。一次中断诊断生成至多 128 MiB 私有临时状态，通过恢复后核对文件哈希，仅删除该文件及其回执；失败时保留用于定位。完整 trace 每路线最多 128 MiB，两路线至多 256 MiB，留在该诊断目录，不自动回传。其他输出是紧凑 JSON、有限监督记录和日志。显存报告使用共享预热 allocator，反映此完整更新的峰值，不代表每条路线独立冷启动最小值。

代码与控制面由 owner 使用标准库/替身、真实 CLI 和 Bash 参数往返验证；owner 不导入 Torch/NumPy，也不运行模型/GPU。插件方案仍为 `approved`，真实 trace/recovery/收益需要师弟执行本次命令后审阅。通过后再决定 Graph 组合与正式训练迁移，原始 checkpoint 不被诊断状态取代。

关联：`attention-update-mechanism-recovery.json`、`attention-update-plugin-plan.json`、`docs/evidence/attention-update-delivery-20261115.json`。
