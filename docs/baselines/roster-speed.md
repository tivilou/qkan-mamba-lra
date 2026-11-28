# Baseline 未完成队列：一键耗时测量

`--all-pending`读取原Suite的实际25项任务和状态，逐项覆盖全部未完成模型×数据集，而不是从截图猜任务数。合法完成结果优先于滞后failed/pending状态，核验后跳过；已经运行过的暂停任务优先复用该任务自己的累计吞吐和实际执行证明。尚无记录的MEGA/S5/S4D任务在独立目录做真实worker短跑。原配置、原固定科学代码和原环境不变，不推进正式断点或恢复正式队列，不安装依赖。不测子集精度，不启动正式全量训练。

## 实验机执行

先在原训练终端按一次Ctrl-C，等待保存并退出。保持GPU空闲；随后执行整段：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
ROSTER_SPEED_REV=$(git rev-parse FETCH_HEAD) &&
ROSTER_SPEED_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${ROSTER_SPEED_REV:0:12}" &&
cd "$ROSTER_SPEED_ROOT" &&
.venv-baselines/bin/python -B scripts/experiment.py diagnose roster-speed -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-89353fc83f89/runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --all-pending --gpu 0 --window-seconds 300 --timeout 2400
```

同一终端显示当前模型×数据集、初始化／预热／测量阶段、样本数、micro/逻辑batch、三个测速窗口、GPU占用和日志位置。全部依次运行，Retrieval排最后，不并跑。失败后仅在进程组清理和GPU空闲确认后继续下一项，一次性回传。不支持的特殊绑定worker若没有自己的历史吞吐，则明确列为缺测，不拿普通未融合模型冒充实际策略，也不把缺测时间计为零。

不加`--all-pending`仍兼容原S5/S4D ListOps双任务入口。此前短测两项未满三窗口，所以未启动的ListOps任务此次会重新测量。已暂停Mamba Retrieval沿用同job的原FP32融合worker心跳和实际融合计数，标注“历史运行估计，非新三窗口测速”；其它有合格运行心跳的任务亦然。历史心跳与配置文件哈希写入报告，不称为当前硬件新实测。

测量从独立随机初始化开始，配置和实际worker派发路径沿用当前Suite。S5复用原JAX环境和已有CUDA动态库／确定性参数，S4D minimal复用原Torch环境。原Suite配置、状态、已有断点指针和完成记录在锁内校验哈希；临时checkpoint只在确认子进程组清理后删除，私有日志保留。

## 时间与结果范围

首个训练心跳后预热至少30秒且5次更新，用worker时间戳及累计样本记录三个窗口。上述指令每个窗口300秒，即有效测量至少15分钟，另加启动和预热。micro/逻辑batch或OOM重试计数变化会重置测量。单项总墙钟上限40分钟，安全保存及进程清理另计。逐项时间依赖真实未完成任务数；例如12个新测任务，有效窗口本身至少约3小时，不代表只花15分钟就全测完。

**原首epoch上限已移除。** 临时worker保持原配置并允许正常跨epoch及原验证；只在整个训练接近最后epoch前设置防误入最终test的保底上限`ceil(train_rows/max_logical_batch)*(epochs-1)`。ListOps100epochs/maxlogical128时从原749步扩为74250步，三个窗口不再因749步截断。窗口内epoch验证和周期保存计入实测推进节奏。仍保留超时、OOM、进程异常、用户主动暂停以及最终测试前保护，不能声称任何故障下均持续运行。没有三个完整窗口时`measurement_complete=false/passed=false`，保留部分估计但不冒充测试完成。

三个完整窗口速度极差不超过15%时标记`stable=true`，否则明确为初步估计。这个15%只描述测速波动，不是模型精度、恢复或正式准入阈值。样本吞吐按窗口总样本／总时间计算；首次心跳墙钟单独列出，包含导入、数据、模型、首次编译及首步/心跳等待，不能解释为纯JIT编译时间。

回传`report.json`和`summary.md`：所有跳过的已完成项、选定项、逐项测量／历史复用来源、吞吐、全量从头耗时、当前剩余耗时，以及覆盖度和已知项合计。全量按各任务自己的训练集规模及配置epochs计算，不把ListOps速度套到长序列或Retrieval。暂停任务剩余样本来自其最后心跳，明确不是解包checkpoint后的精确进度；有checkpoint但无可用心跳时剩余时间写unknown。缺测和失败不作为0小时计入“全部完成”的合计；`coverage_complete`明确标识覆盖是否完整。

同时保留batch/OOM事件、首心跳墙钟、安全保存清理、5秒采样整卡峰值显存等。吞吐外推含窗口内正常epoch验证及周期保存，不含启动、最终验证/test/trace/回传或尚未观察到的长程开销，所以称大致执行节奏而非保证墙钟。1/10仅工作量外推，不是子集训练精度；窗口范围不是统计置信区间。

窗口内若发生周期checkpoint保存，其耗时自然计入该窗口的训练推进速度；最后一次安全暂停保存／清理单独计时，不混作稳定训练窗口。短跑仍可能低估长程执行中尚未出现的开销。

统一回传位置：`qkan-mamba/roster-speed/<运行名>/`。传输失败保留本地报告和原回传清单；重传不会重新测速：

```bash
.venv-baselines/bin/python -B scripts/experiment.py diagnose roster-speed -- \
  --return-only /root/projects/CONTROL_CHECKOUT/runs/roster-speed-RUN_ID
```

替换为终端实际打印的本次Suite绝对路径。测速结束后正式队列**保持暂停**，供作者决定实验规模。需要恢复原实验时，回到MEGA策略首次激活的`mega-policy.json`所记`code_root`目录，使用原一体化resume命令；不从本次新测速checkout替换该绑定。
