# 64步试跑：只重试最后一项 Graph32 比较

[认证审阅](../evidence/key-bias-stability-partial-review-20261117.json)已核对5d7ec8f执行的246354字节报告及22项代码/计划绑定。原FP32、FP32解析、BF16解析各64步的8步块恶化门通过，BF16连续64与新进程32+32的更新记录、九类完整状态哈希精确一致。Graph16中位完整更新7.9384835511446秒，BF16解析3.782439388334751秒，吞吐约2.0987735倍，四个匹配对均为正。该证据限于冻结训练前缀与单批计时，仍无完整收敛、验证/测试效用或Graph32比较结论。

最后的partition32子进程以AccelerationRequired退出；七个子进程均已回收且原checkpoint/元数据未变。旧父进程以worker-error.json覆盖comparison.json，比较入口也未导出Executor内部拒绝原因，因此现有报告只证明最后一项失败；显存、编译预算、数值或其他原生原因尚待区分。进度标记到计时index6后，在最后一次路线准备时退出；旧报告未返回可复算的部分计时。

本次修复仅补充诊断与断点复用：逐条保留已完成计时，清理前保存执行器内部结构化错误类别/代码位置或数值不一致字段；同时回传comparison.json与worker-error.json。保持原微批、门限、预算和执行器，错误继续停止，不以降级结果作为Graph计时。修复不宣称已解决尚未观测的Graph32根因。

## 执行

原训练须保持暂停、GPU0空闲；在tmux中执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_RETRY_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_RETRY_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_stability \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --screen-from /root/projects/qkan-mamba-lra-adaptive-91b8df662c39/runs/key-bias-20261116T090926Z-60f6f2a2 \
  --benchmark-from /root/projects/qkan-mamba-lra-adaptive-b3a337d4267f/runs/key-bias-benchmark-20261116T124533Z-26d47029 \
  --reuse-from /root/projects/qkan-mamba-lra-adaptive-5d7ec8f0cb9f/runs/key-bias-stability-20261117T003301Z-fd47ab92 \
  --child-timeout 900 --stall-timeout 300
```

保留原screen、benchmark及其引用目录和失败Suite中的stability-input.json/key-bias-stability.json，复用既有本地环境/缓存/账户私密交换配置。新入口核对原报告SHA256 `5b7953f892aad062e80134d94733ecd2d0c484327087b1e173bdc5c8391517d1`、原请求SHA256 `f8c111c7c448940ef1334fbdf87ddc5fd7debe934a955be5c66150d52ba4d548`、原起点/元数据/配置/策略、未改动执行引擎与当前实现绑定，并重算六项结果的门、恢复和Graph16计时/驻留证明；任一不符便停止。不会重新运行六项前缀，也不重建已删除的32步中点。原正式源若推进，保留最新状态并停止，不回退迁就诊断。

新Suite只创建partition32子进程，900秒总上限/300秒阶段上限不变；最终报告将六项已审阅结果与一个新结果明确分开记录出处，并自动回传原认证key-bias-stability命名空间。失败也回传紧凑原因、已完成计时、回滚和自建组清理结果，供下一次定位。诊断只在副本临时更新，退出恢复完整状态；正式提交样本/更新仍为0，候选准入关闭。没有新增正式训练命令。

## 本地验证与回滚

471项无原生框架导入的控制测试：468通过、3跳过；新增5项覆盖报告/请求/源/执行引擎篡改拒绝、门与恢复重算、单partition32调度、两份失败材料同时留存，以及真实stdlib Executor的内部拒绝/结构化数值字段在降级后的保留。模拟第三次捕获拒绝时，旧入口返回0条部分计时且无内部原因，修复入口保留7条计时与具体类别；两者完整状态回滚均精确。模拟结果不代表GPU故障根因。

可审查包在`.runtime/key-bias-stability-fix-20261117/`：candidate/、change.patch、verification.json、rollback.py。verification记录基线/修复的逐字命令、输入、输出与退出码；独立副本验证补丁、回滚与重放，八个正式核心和已执行计划保持原哈希。回滚命令为`QT2/python.exe -B rollback.py <待回滚副本绝对路径>`，仅恢复包内五个交付文件，不触碰训练目录。
