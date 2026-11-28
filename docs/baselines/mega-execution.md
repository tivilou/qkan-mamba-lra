# MEGA 可选执行插件与一次联合验证

已接入现有 `execution_plugins` 注册表和原 worker 的 `Hooks.model_context`，插件为 `mega.fft_tail_bf16` v1。核心训练、损失、Adam、样本时钟、保存和恢复代码保持原样。

```json
{"plugin":"mega.fft_tail_bf16","version":1,"graph":false}
```

`graph` 为独立布尔开关，省略时默认关闭。训练时启用 FFT 补齐及最后一层 FFN Linear 的 BF16，其输出返回 FP32；主参数、Adam及其余路径保持FP32。验证／测试使用原 FP32 路径。局部 Graph 缓存仅为临时运行状态，输入形状变化会重新捕获，保存恢复不会序列化旧图。

注册表支持代码适配的数据集，不代表每个数据集已经验证。此次仅对自动选中的一个未完成 MEGA 任务取最新完整断点，进行一次有界联合验证。已有 Text 单batch测速不能替代其他任务验收。

## 已审阅结果与正式策略切换（2026-11-27）

### 2026-11-28：训练完成后的回传导入修复

实机截图：MEGA Text已到epoch100/step15697，`summary.json`已写出，但`mega_formal.native`最后回传凭证时导入`return_artifacts`，其`from scripts.exchange_artifacts ...`报ModuleNotFoundError。原进程已退出。这是回传阶段包解析冲突，不是梯度更新报错；后续解释器关闭时的`official_source`生成器异常是次级清理现象，不作为主故障。结果完整性仍由原完成结果验证器核对，不只根据文件存在认定成功。

新控制器仅用`scripts/pinned_mega_worker.py`在上游导入前按绝对文件路径预加载**原执行器自己的**`scripts.exchange_artifacts`子模块；不替换上游` scripts`父包、不改原worker文件或绑定，不调整精度或样本。新控制器从既有`mega-policy.json`读取原`code_root`的策略helper，继续验证该旧版本和原科学哈希，避免换了UI/回传修复checkout后错误要求迁移科学版本。`command.json`记录兼容包装器绝对路径和SHA。

当前已退出、GPU空闲时执行整段恢复。不要重复`--mega-policy`激活、删除summary或回退旧step：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
MEGA_REPAIR_REV=$(git rev-parse FETCH_HEAD) &&
MEGA_REPAIR_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MEGA_REPAIR_REV:0:12}" &&
cd "$MEGA_REPAIR_ROOT" &&
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last
```

该指令恢复正式队列并显示原看板。原worker读取最新完整state；若`training_complete=True`，跳过训练更新，只恢复既有最佳权重的最终评估/trace、保存结果、回传及原结果校验。当前错误不要求重训100epochs。完成后继续排队任务。若要先跑S5/S4D测速，则先保持队列暂停、执行`roster-speed`，不要同时占用GPU。

代码回退只需使用此前原控制器入口（保留其回传冲突限制）；新包装器不会修改原科学文件或现有状态，不用修改/伪造旧绑定。传输仍沿用原凭证和自动回传清单。

实际worker联合验证已通过：匹配8步、同一路线4＋新进程4步恢复状态哈希完全一致。FFT＋尾层BF16无Graph吞吐增加11.41%；加入局部Graph后增加11.20%，所以本次Graph关闭。固定验证集前256行从217行正确变为216行，尚未证明全量精度或收敛等价。用户已明确选择切换为这一精度变体，记录于 `mega-text-policy-switch-20261127.json`，不改写旧资格证或声称原FP32等价。

仅切换 **MEGA Text**；其他模型及MEGA其他数据集继续原派发。原Suite、配置、完成结果及Retrieval最后的顺序保留。新控制器通过原派发函数的私有副本替换选定job的spec和结果检查，不替换模块全局。新worker复用已验证的稳定worker及Hooks；原代码单独验证旧certificate，新amendment记录新插件来源。激活前在Suite事件目录保存原始完整generation及指针；继续训练始终读取当时最新完整断点，不回到测速6016步。

当前训练若仍在跑，先在原终端按一次Ctrl-C，等待保存并退出；然后执行整段，训练与美化看板在同一个终端：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
MEGA_POLICY_REV=$(git rev-parse FETCH_HEAD) &&
MEGA_POLICY_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MEGA_POLICY_REV:0:12}" &&
cd "$MEGA_POLICY_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last --mega-policy fft-tail-bf16
```

不再重复联合测速。启动后看板应显示 `MEGA策略：FFT补齐＋尾层 FFN BF16 ... Graph关闭 ... 实际调用已确认`；只有当前PID、时间、amendment/mode及三种调用计数全部匹配才显示确认。未出现时查看该job的 `worker.log`、`mega-policy-error.json`。每次暂停／完成自动回传紧凑执行凭证至 `qkan-mamba/mega-policy/`，失败则保留本地重传文件。新的summary和environment携带策略来源；历史正式配置及旧结果不被重标注为已加速。

后续在**首次切换使用的控制器目录**继续执行上述python命令即可；普通resume会自动沿用已记录策略。原目录保留旧代码，直接从旧目录启动旧脚本不会自动启用新插件。若换到其他新checkout，应回到Suite `mega-policy.json` 的 `code_root`，保持明确绑定。

回退时先等待安全暂停，仍在同一控制器目录执行下列单条指令。它关闭插件、继续最新状态，保留已经执行过的BF16阶段及其来源，不抹去训练进度或声称消除既有数值影响：

```bash
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last --mega-policy original
```

## 历史联合验证入口（不要为切换策略重复运行）

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
