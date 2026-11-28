# S5 / S4D minimal：ListOps 一键耗时测量

只测尚待排期决策的两个模型，分别执行现有正式 worker 的真实训练更新。独立输出目录、原配置、原固定科学代码和已安装环境；不从正式断点推进、不安装依赖、不改变正式队列。不测试精度、不启动全量或1/10子集训练。

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
  --gpu 0
```

同一终端显示当前模型、初始化／预热／测量阶段、样本数、micro/逻辑batch、三个测速窗口、GPU占用和日志位置。S5然后S4D minimal依次运行，不并跑，不启动正式训练。一个失败且进程组清理/GPU空闲确认后仍尝试另一个，统一回传；不重复失败路线。

测量从独立随机初始化开始，配置和实际worker派发路径沿用当前Suite。S5复用原JAX环境和已有CUDA动态库／确定性参数，S4D minimal复用原Torch环境。原Suite配置、状态、已有断点指针和完成记录在锁内校验哈希；临时checkpoint只在确认子进程组清理后删除，私有日志保留。

## 时间与结果范围

默认首个训练心跳后预热至少30秒且5次更新，然后用worker时间戳及累计样本记录三个至少60秒窗口。micro/逻辑batch或OOM重试计数变化会重置测量。每个模型总墙钟上限20分钟，安全保存及进程清理另计。也会在首个epoch验证前按最大逻辑batch推导的步数上限自动暂停，防止为了测速跑到验证／测试。过快触及上限、编译／调档过久时仅返回已有短窗口和“估时暂不稳定”，不为了凑窗口修改训练预算或跨入评估。

三个完整窗口速度极差不超过15%时标记`stable=true`，否则明确为初步估计。这个15%只描述测速波动，不是模型精度、恢复或正式准入阈值。样本吞吐按窗口总样本／总时间计算；首次心跳墙钟单独列出，包含导入、数据、模型、首次编译及首步/心跳等待，不能解释为纯JIT编译时间。

回传`report.json`和`summary.md`，包括：实际配置来源、窗口与吞吐、batch／OOM事件、初始化到首心跳时间、安全保存清理时间、5秒采样的整卡峰值显存，以及全量/1/10训练工作量估时。显存不是框架分配器的精确高水位。ListOps当前96000训练样本；按原配置100epochs计算，全量960万样本次、1/10为96万。估时不包含初始化、验证／测试、最终回传等开销，窗口区间不是统计置信区间；1/10是工作量外推，不是已经实际训练过子集或测得子集精度。

窗口内若发生周期checkpoint保存，其耗时自然计入该窗口的训练推进速度；最后一次安全暂停保存／清理单独计时，不混作稳定训练窗口。短跑仍可能低估长程执行中尚未出现的开销。

统一回传位置：`qkan-mamba/roster-speed/<运行名>/`。传输失败保留本地报告和原回传清单；重传不会重新测速：

```bash
.venv-baselines/bin/python -B scripts/experiment.py diagnose roster-speed -- \
  --return-only /root/projects/CONTROL_CHECKOUT/runs/roster-speed-RUN_ID
```

替换为终端实际打印的本次Suite绝对路径。测速结束后正式队列**保持暂停**，供作者决定实验规模。需要恢复原实验时，回到MEGA策略首次激活的`mega-policy.json`所记`code_root`目录，使用原一体化resume命令；不从本次新测速checkout替换该绑定。
