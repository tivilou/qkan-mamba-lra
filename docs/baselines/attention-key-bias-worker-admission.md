# 最终 worker 的有界准入

状态：b23e1da的实际worker原生准入已完成并通过认证审阅；正式队列尚未启动。四个子进程均退出0，连续2步与新进程1＋1步的26项科学持久化字段、更新记录及参数梯度精确一致，固定验证行0、1的两路线真实轨迹通过。原checkpoint和两项完成结果未变，8份自有诊断checkpoint已删除。见[正式审阅](../evidence/key-bias-worker-native-review-20261117.json)。[冻结计划](attention-key-bias-worker-plan.json)保留执行时原文，其中待验证描述由本次证据更新，不改写历史绑定。此前数值、64步稳定性及两组性能结果沿用原审阅，不重复运行。

当前下一步是交付单独的正式迁移／长期启动入口，绑定已审阅的worker执行文件、最新兼容原源及完成结果；现有有限准入入口和常规worker CLI均没有开启候选长期训练的路径。此次通过不等于正式续跑已经开始，也不等于完整收敛或验证／测试效用通过。下面保留已执行的准入协议和命令用于追溯，本次无需重跑。

2026-11-17重试说明：首次09b0382运行在迁移准备前退出，原因是监督器预建`output/diagnostics`与入口的“目录必须不存在”检查冲突。现复用既有`prepare_worker_output`：只允许监督诊断目录预先存在，旧checkpoint、配置、结果或链接仍拒绝。新增真实监督器启动控制；未修改worker更新主体、冻结协议或既有成功实验。首次回传与来源保存审阅见[故障审阅](../evidence/key-bias-worker-bootstrap-native-review-20261117.json)。下面命令先准备新的干净checkout，再重做尚未执行的四个准入子进程。

第二次7829579运行已通过目录检查、模型准备和零更新checkpoint保存／重读比较，在写`migration-validation.json`时发生重复`target_contract`关键字错误。现只修正回执字典合并顺序，保留所有状态比较、原元数据及0更新约束；新增整段生产迁移分支的标准库替身测试，覆盖候选、旧路线、损坏状态拒绝及回执字段优先级。原源未变，两份自有step14268诊断代际已清理；四项完整准入仍待重试。见[第二次故障审阅](../evidence/key-bias-worker-migration-receipt-native-review-20261117.json)。

本次仅扩展 `worker.run` 的显式策略分支。未选择时保持原数值更新主体；选择后使用原实例适配器和解析K段hook，固定eager16／logical128。候选训练和评估均用BF16注意力，FP32主参数、投影、残差、head、loss及原Adam历史保持。与Graph、其他模型任务及未知策略组合均拒绝。常规worker CLI不具备启用权限，新的准入入口只允许有限步数。

完整checkpoint增加策略版本、精度、梯度策略、配置、原源checkpoint和执行文件绑定，以及全部参数梯度。已有候选checkpoint须严格匹配；原FP32源的首次转换单独记录，移除旧进程加速状态，不重置参数、矩、step、衰减、样本时钟或排列。原源没有保存梯度，因此初次转换明确设为None；后续候选checkpoint实际保存并恢复全部梯度。

新入口先检查空闲GPU、源锁、原最新checkpoint与两项已完成结果，再复制保留结果。它运行四个独立进程：零更新迁移验证、连续2步、中断1步、新进程恢复1步。连续与中断都从实际零更新验证后的副本开始。总计4次副本更新、512个处理样本，仅256个不同原训练行；每子进程900秒、阶段停滞300秒。若原源推进，则保留新进度并绑定最新源；若当前epoch不足256行则退出，待单独处理，不回退到旧14268。

连续2步与1＋1步比较所有科学持久化字段、每次更新记录及参数梯度；明确排除训练用时、控制器计时／显存遥测、峰值字节、进程加速记录与回滚设备。每次更新重新固定物理／逻辑批量。这是科学状态精确恢复，未宣称整个checkpoint文件或计时遥测逐字节相同。

恢复后的实际worker模型生成固定验证行0、1的FP32／BF16配对 `sample-trace.v2`：同一暂停权重、输入和标签，完整私有数组重读、坐标访问、结构／语义验证、独立池化与head重放。checkpoint标为 `latest_preflight`，而非best-validation。该验证只检查最终worker前向，不重复此前真实训练梯度轨迹，也不报告验证集效用或最终模型案例。私有值总额上限128MiB，轨迹前后检查模型、梯度和RNG不变。

无论成功或失败，入口都会自动回传紧凑 `key-bias-worker-admission.json`，使用既有认证交换及SHA256回执；原始值、日志与权重留在本机。比较结束删除自有诊断checkpoint代际，保留紧凑回执、两个完成结果副本和固定案例值。续训源始终是原始完整源，不使用诊断中点。

合作者在最新已发布checkout中执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_worker_admission \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --pilot-from /root/projects/qkan-mamba-lra-adaptive-5fe863648d4c/runs/key-bias-stability-20261117T050617Z-0fc6b309 \
  --child-timeout 900 --stall-timeout 300
```

执行后审阅新回传报告，才决定后续正式迁移。此次命令不启动25任务队列，准入通过也不等于完整收敛或科学性能改善。
