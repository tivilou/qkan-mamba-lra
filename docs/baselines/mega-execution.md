# MEGA 可选执行插件与一次联合验证

已接入现有 `execution_plugins` 注册表和原 worker 的 `Hooks.model_context`，插件为 `mega.fft_tail_bf16` v1。核心训练、损失、Adam、样本时钟、保存和恢复代码保持原样。

```json
{"plugin":"mega.fft_tail_bf16","version":1,"graph":false}
```

`graph` 为独立布尔开关，省略时默认关闭。训练时启用 FFT 补齐及最后一层 FFN Linear 的 BF16，其输出返回 FP32；主参数、Adam及其余路径保持FP32。验证／测试使用原 FP32 路径。局部 Graph 缓存仅为临时运行状态，输入形状变化会重新捕获，保存恢复不会序列化旧图。

注册表支持代码适配的数据集，不代表每个数据集已经验证。此次仅对自动选中的一个未完成 MEGA 任务取最新完整断点，进行一次有界联合验证。已有 Text 单batch测速不能替代其他任务验收。

## 实验机执行

原训练终端先按一次 Ctrl-C，等保存并退出，再执行整段。已暂停则直接执行。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
MEGA_DIAG_REV=$(git rev-parse FETCH_HEAD) &&
MEGA_DIAG_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MEGA_DIAG_REV:0:12}" &&
cd "$MEGA_DIAG_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mega-qualify -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-89353fc83f89/runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0 --job auto &&
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last
```

最多7个独立子进程、40次临时更新、0次正式更新。以最新断点的当前epoch剩余完整batch为准，统一选择不超过8的最大偶数步数（2／4／6／8），所有对照使用同一步数；Graph关、开各在一半处保存，新进程恢复另一半。报告标明实际 `planned_steps`，缩短窗口不视为完整8步证据。若不足2个完整batch，明确零更新跳过；原队列正常继续，不推进源断点、不跨epoch绕过该检查。全部使用原worker真实更新／checkpoint，不另造简化恢复器。基线使用相同断点和样本前缀，临时禁用吞吐调参；每步核对实际micro和逻辑batch，OOM降档后不认定为匹配比较。恢复沿用现有 `rtol=1e-5, atol=1e-6` 比较、RNG和非浮点控制精确一致；不放宽恢复容差。

连续8步的三条路线额外用原FP32评估固定的验证集前256行，报告loss／accuracy／样本数和每步训练loss。无测试集访问、不用该片段重新挑选训练检查点、只观测一次。速度包含实际worker数据搬运、更新和保存等执行成本，Graph开关直接在相同起点对照。

结果统一回传 `qkan-mamba/mega-qualification/<运行名>/qualification.json`。报告中的 `technical_passed` 表示执行、匹配条件和恢复检查通过；`quality_admitted` 保持false，精度效果需根据实际验证证据判断。现有协议没有针对该组合冻结的精度准入阈值，不能把“恢复通过”替代“精度无损”，也不新增宽松阈值自动切换正式任务。

联合结果回传前，原Suite和正式策略保持原样，整段结束恢复原训练／美化看板。正式队列迁移和启用仍以联合结果审阅为前提，本次未改旧Suite绑定或重跑已完成任务。回传失败保留报告及重传清单，不重新测速；进程清理或源哈希检查异常会阻止自动续跑。


2026-11-27入口修复：原版本训练前错误回传不完整，父进程通用失败覆盖了子进程原始位置，精确历史原因不能从旧报告确认。现在子进程主入口捕获准备阶段的全部异常并记录调用位置／过滤错误链，父错误单独保存于 `parent-errors`，回传保留native/detail/setup。参考路线在共同准备阶段失败后不再派发后续候选，避免重复相同错误。固定8步条件已改为上述匹配窗口；原始源配置／完整检查点和恢复容差不变。
