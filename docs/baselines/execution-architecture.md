# 实验执行结构与扩展边界

本次收敛范围是 `experiments/baseline_delivery` 的执行链与统一命令入口。
历史 BF16/Graph 的固定版本、实验结果、原运行目录和旧入口保留兼容；不批量搬迁项目文件。

| 层 | 现有实现 | 职责 |
|---|---|---|
| 用户入口 | `scripts/experiment.py` | start/resume/watch/upload/diagnose；同屏监督与看板 |
| 稳定训练核心 | `worker.py` | 原训练/验证循环、Adam、采样时钟、best/latest、保存恢复 |
| 任务适配 | `data.py`、`models.py` | 数据身份、输入结构、原模型与指标 |
| 执行插件 | `execution_plugins.py`、`mamba_convolution.py` | 显式注册、组合检查、实例级算子适配 |
| 实验控制 | 既有 formal/diagnostic 模块 | 固定任务单、来源、有限测试、结果回传；复用原核心能力 |
| 显示 | `scripts/formal_progress.py` | 读取状态、呈现实际后端/精度/吞吐；不决定科学策略 |

## 一次接口升级，后续通过插件扩展

`worker.run(..., execution=None)` 保持原参数与默认路径，新增一个可选参数。
`execution_plugins.Hooks` 提供模型上下文、执行器恢复状态、控制器调优历史、
保存前后和实际心跳发布的明确扩展点；不接管训练/恢复循环。
原梯度计算、损失、更新、checkpoint读写及RNG函数的AST与89353fc逐项对照保持一致。
这属于有限的核心接口升级，不声称核心文件从此永不修改。

普通入口不传 `execution` 时使用原路径。插件通过固定白名单创建，禁止配置任意导入路径；
当前正式接入候选仅 `mamba.fused_fp32` v2，FP32/eager，Mamba Retrieval/Image/Pathfinder。
BF16与Graph尚未注册为该插件的可组合模块，不能据此声称任意组合已经可用。

v2不再替换训练器的 Classifier、Executor、save_checkpoint 或 atomic_json。
原Mamba forward使用实例私有全局命名空间，融合autograd定义来自固定SHA源码，
仅在插件私有命名空间注入已验证的官方CUDA ABI桥；训练融合、验证普通FP32。
原模块全局对象、其他模型实例、参数对象和state_dict键不变；退出恢复实例方法。
历史诊断使用的旧FusedFP32保留原入口，不作为新插件实现复用。

## 配置、谱系和显示

v2请求单记录请求/解析的插件标识、版本、FP32/eager边界，实际调用数仍写现有执行证明。
新科学绑定覆盖训练核心、任务和执行依赖，排除统一入口与显示脚本。
显示修复可以通过既有 `.bootstrap` 控制脚本方式连接原固定执行器；不重写旧绑定或旧收据。
每个实验仍固定自己的执行checkout，恢复沿用该实验配置与实际策略。

v1→v2是显式新分支，从暂停源的**最新完整检查点**复制并保留已完成结果，禁止退回24981旧断点。
新的执行机制清空旧吞吐测量窗口/基线，保留当前micro/logical档位、OOM记录和科学进度。
正式v2仍通过既有连续2步与新进程1+1检查后才允许长跑；本地fixture不是服务器验证。

## micro测试

复用 `experiment.py diagnose mamba-speed -- --micro-benchmark`。
一个监督子进程，逻辑batch固定32，micro8/16/32各预热2次，再按8/16/32/32/16/8交错两轮，
共18次临时更新尝试。每次从相同最新断点、相同32行恢复；不提交正式更新。
只捕获明确CUDA OOM，恢复状态并标记该档；其他错误直接回传失败。
输出每档完整更新耗时、吞吐、峰值allocated/reserved显存、有限loss及实际融合调用。
推荐档需比micro8中位吞吐提升至少3%，且4组成对测量至少3组更快；否则保留8。

耗时包含CPU取数/H2D/前向/反向/裁剪/Adam/调度/同步，排除恢复、预热及正式worker事务快照开销。
它是固定一批样本的执行测速，不是整数据集吞吐、恢复认证或数值等价认证。
micro变化可能改变dropout和归约，测速结果不自动修改正式档位。
源自适应最近32条标量事件同时回传，帮助区分物理扩档、逻辑扩档与历史回退。

测试后删除自有检查点副本，保留紧凑报告、回传收据与日志；原训练目录保持原样。
师弟的运行与返回指令见 `mamba-validation.md` 当前测试段。

## 显存换速度的有限测试

`mamba_memory_benchmark.py` 是诊断适配器，复用原 `mamba_diagnostic.native` 的
完整更新回调；稳定worker和正式保存恢复不修改。实例插件仅增加显式checkpoint级别设置，
默认仍为1，候选0通过私有融合函数参数传入，不替换全局对象。
CPU lookahead和锁页staging两侧共用；GPU双缓冲用独立复制流，复制/计算事件保护槽位及
主机buffer复用。Graph使用独立Module包装，共享同一模型参数但不改eager对照的forward。
各方法独立子进程，最多44次临时更新；仅独立有收益且RNG/时钟一致的至少两项再测组合。
退出0表示整批尝试及清理回传完成，单项失败显式记录；不表示候选已准入。
