# 解析K梯度：短轨迹与完整更新测量

前置证据为[数值筛查](../evidence/key-bias-native-review-20261116.json)和[新策略恢复/真实轨迹审阅](../evidence/key-bias-recovery-native-review-20261116.json)。两次已完成的实验不重复。新入口绑定两份原报告和请求、原checkpoint、当前实现与[独立计划](attention-key-bias-benchmark-plan.json)，只运行一个受监督的诊断子进程。

## 冻结计算与判断

- 三路线：原FP32、FP32解析K梯度、BF16解析K梯度。每条从原step14268完整状态起步，micro16/logical128，固定接下来的4批，共512条训练行，dropout0。各步样本哈希必须相同；两个解析路线逐步CE绝对差异相对原FP32均须<=0.10，且真实运行时检查loss/梯度/参数有限性。未通过即停止profile和计时。这是短轨迹筛查，不是收敛结果。
- 原FP32与BF16解析各一次完整更新profile，使用原dropout0.1，必须出现efficient-SDPA前向与反向ATen算子，排除flash/cudnn/math替代。记录共享分配器的完整更新allocated/reserved峰值，不据此宣称独立最小显存或未来训练余量。
- 两路线各2次预热，然后按`A B B A A B B A`计时，A=原FP32、B=BF16解析，各4个同步完整更新。每次都从同一完整状态恢复，数据选择/H2D/前向/真实损失/反向/解析hook/裁剪/Adam/样本调度/有限性检查在计时内；恢复、预热、profile、计时后的完整状态哈希在计时外。原dropout启用，同路线4次更新输出及完整状态须精确一致。
- 速度成功门：中位完整更新吞吐收益>0，且4个邻接匹配A/B对的收益均>0。单对非正即保留测量并标记gate_rejected，不进入正式训练。这只是最小诊断门，4对不能证明统计显著性或长期收益。
- 总计26次临时更新：短轨迹12、profile2、预热4、计时8。单GPU预算通常数分钟，900秒总上限、300秒不变阶段上限；原训练需按既有流程暂停，GPU空闲。无epoch循环、验证/测试访问、数据下载、安装或正式训练提交。

候选仍只改K偏置梯度定义，保留全部53参数、807170元素和Adam历史；全局数值门限及训练核心保持原样。复用完整更新运行时，恢复入口新增显式operation注入供本诊断调用；不改变默认恢复分支，不作全局替换。已执行旧插件计划与旧报告保持不变；新计划是本阶段扩展，保持approved和正式关闭，实际benchmark尚待合作者回传。

## 执行

原训练按既有流程暂停、GPU0空闲后，在tmux执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_BENCH_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_BENCH_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_benchmark \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --screen-from /root/projects/qkan-mamba-lra-adaptive-91b8df662c39/runs/key-bias-20261116T090926Z-60f6f2a2 \
  --preflight-from /root/projects/qkan-mamba-lra-adaptive-33ee6dd8c6b8/runs/key-bias-recovery-20261116T120747Z-5a5247d4 \
  --child-timeout 900 --stall-timeout 300
```

依赖保留原screen及其归因目录、preflight报告、原数据/cache/checkpoint。源若已推进并与审阅14268不符，停止并保留当前进度；不回退正式训练来迁就诊断。父进程持有原supervisor/suite/worker锁，仅清理自己创建的子进程组。

## 回传与回滚

自动回传`key-bias-benchmark.json`到认证交换`qkan-mamba/key-bias-benchmark/<Suite>/`并核对SHA256回执。报告包含执行绑定、子进程清理、短轨迹、profile/显存、全部计时与回滚；失败也保存部分结果与代码位置。私有原数据、模型、完整轨迹不上传；本轮复用已有真实轨迹就绪证据，不伪造新fixture。

上传失败保留报告与pending manifest，仅重试上传：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory /absolute/path/to/runs/key-bias-benchmark-RUN_ID \
  --experiment key-bias-benchmark --file key-bias-benchmark.json
```

无论通过或拒绝，finally恢复模型/梯度/Adam/调度/进度/排列/RNG，移除hook和实例策略，验证原checkpoint与元数据未变。提交样本/更新为0，诊断状态不成为正式训练来源。后续仍须审阅实际测量，再决定收敛与接入。
