# Torch 编译、CUDA Graph 与真实断点校准

2026-11-10 自动回传`failure-6c49066ce7ea`确认：v5 Graph通过真实数值/RNG检查并回放96次，但普通执行8.833338秒/更新、Graph8.863159秒，慢约0.338%。独立阶段诊断中反向占普通执行记录CUDA时间约75.06%。这已足够转向算子归因，不再用相同候选反复校准。原840训练源保持`runs/adaptive-migrated-20261109T045101Z-183941`、Retrieval step12683及两个原9a完成结果。准备Suite没有提交更新或启动正式训练。详见`docs/evidence/compiler-phase-profile-20261110.json`；不能将Graph可回放称为已有加速。

## 算子耗时专用采集

`scripts/collect_operator_profile.sh`从原840的认证快照运行一次完整的普通执行更新，使用已有逻辑batch和micro-batch，不编译、不测速、不重跑25任务验收，也不续跑正式训练。开始前验证代码/策略、原准入证据、checkpoint哈希、数据契约和运行时；原监督器或worker仍存活时由原锁拒绝采集，不自动中断它们。全部输出进入新的`runs/operator-profile-*`；两个完成任务和原训练断点不改写。

仅在这次更新启用PyTorch CPU/CUDA profiler，以固定范围标记前向/损失、反向、裁剪/优化器/调度。每阶段输出GPU自耗时排名前12的CPU算子、调用次数、主机自耗时、其余耗时合计和未归因项。CPU算子的`self_device_time_total`减去子算子CUDA时间；不再累计对应CUDA设备事件，避免双算。GPU耗时是关联kernel的时间和，可能跨流重叠，并非墙钟关键路径。未找到CPU父范围的事件保留为`unattributed`，不猜测属于反向。

不采集shape、内存轨迹、栈、FLOPs、源码、输入或张量值，不导出Chrome trace；未知/过长算子名归为固定标签。最多检查100000条事件，超限不返回有偏的部分排名。原始事件只在该子进程内存中暂存，JSON只保留有界摘要。首次更新可能含CUDA惰性初始化，采集有开销，不能用这次墙钟与之前测速比较。此入口不测Graph的kernel归因，也不宣称加速。

结束后用原精确标准验证模型、Adam、调度和RNG回滚，再声明0样本/0更新提交；采集器不可用、没有CUDA归因或CPU-only时明确标记，不伪造GPU时间。更新异常不被采集器吞掉或重试；回滚不精确就失败，不返回成功。安全摘要随既有SHA-256回执机制自动发往`qkan-mamba/operator-profile/<本次Suite>/operator-profile.json`。若返回网络失败，保留本地报告和delivery manifest，只重传文件，不重跑采集。

普通Graph校准仍保留原两次阶段诊断；当正确但无收益时，现在再加这一次普通执行算子观察，结果嵌入原failure.json的`benchmark.no_gain_operator_profile`。这次观察和原两次阶段诊断均不计入准入测速。专用采集入口优先用于当前已知瓶颈；不用再次跑完整校准来获得相同信息。

专用交付指令见下方；原生profiler是否可用及算子排名仍待师弟RTX3090回传。本机只用标准库/模拟验证。

## v5：校准回滚复用与收益诊断

上一版每轮校准回滚清空CUDA缓存，再经CPU复制重建Adam。v5先检查完整优化器状态结构、参数顺序及张量形状/类型；一致时把备份值复制回原有Adam存储，并恢复原参数组控制值。不会把备份张量直接安装到优化器，也不会与备份共享可变存储。CPU备份可复制回原GPU矩；新建/惰性状态等结构不匹配时，仍走原有独立加载路径，真实OOM仍保留CPU回退。模型/Adam/RNG和完整回滚标准不变。这主要减少校准准备开销并保留热分配器状态，尚未证明正式训练每步因此更快。

测速仍为一次预热和两个测量对；每次恢复之后先同步GPU，再开始完整更新计时，结束再同步，排除队列中未完成的回滚复制。两个测量对按普通→Graph、Graph→普通交换顺序，记录每次原始耗时、执行次序、中央值及收益比例。不会重测直到碰巧得到正收益，也不放宽正收益要求。两个短测量对不构成长期或统计显著性证明。

若正确性通过但仍无收益，保留Graph，在同一完整快照上额外剖析一次普通更新和一次候选更新，分别回滚；不访问验证/测试，不提交训练更新。只回传前向/损失、反向、裁剪/优化器/调度的主机与CUDA流计时、更新总耗时和分配器数值摘要。这两次诊断不计入准入测速，峰值显存标明覆盖整个外层更新。报告自动随原failure.json返回；终端明确显示“数值/RNG已通过，但无收益”，避免只看到泛化的加速失败。

纯Graph主要减少提交开销；若实际瓶颈是长序列的GPU计算，本身没有必然提速。只有新的阶段证据才能支持后续具体优化；本版不承诺原生Graph通过收益门槛或大幅加速。

## v4：Retrieval分图与随机数容量

固定PyTorch v2.7.1源码表明，CUDA Graph内Philox偏移只有uint32容量。原生efficient/cuDNN注意力dropout每层保留`B×heads×Lq×Lk`计数；Retrieval两个4000-token文档、4层4头、micro16，仅注意力就需8,192,000,000，超过4,294,967,295。合成micro4则可容纳。最新回传的错误阶段、位置及RNG分类与此吻合，但旧诊断没有异常原文，所以这是源码支持的主要原因推断，不将其写成已确认的原始异常。

Graph/Dynamo原生反向候选现在分别捕获两个文档的同一个`encode`，使用两个独立图池，维持两个文档的激活直至共享损失反向。共享参数、先文档1后文档2的顺序、原有`abs(a-b)`/`a*b`配对及分类头均保留；配对和分类头在原生Torch执行。没有更换注意力、关闭dropout或改变精度。编译仍为`torch.compile(backend="eager", fullgraph=True)`加显式前向/反向Graph，不能宣称Inductor融合。

捕获前按实际形状检查保守RNG预算：每文档`layers×B×(heads×L²+6×width×L)`，包含原生前馈与残差dropout的元素数上界。当前配置micro16每图上界4,292,608,000，可容纳；micro32超限，在任何捕获前拒绝。专用容量事件通过完整更新回滚只降低物理micro-batch，保持同一逻辑batch及样本，不计为OOM，不因该事件降低下一逻辑batch。确定的容量档位上限持久化，冷却后也不再重复试探超限档位；真实OOM仍走独立原流程。进度把“Graph容量降档”和OOM重试分别显示。

图池生命周期改变必须重新通过精确跨进程恢复、真实同次更新数值检查和完整更新测速；既有容差及精确RNG/控制标准不变。Graph要求下若无正收益或无法通过，停止并自动回传，不能由这次源码推断或模拟测试宣称成功。

## v3修正

候选顺序现在为Graph/Dynamo原生反向、Graph/Inductor、Graph/AOT原生算子、compile/Inductor、compile/AOT原生算子、eager。每个候选独立进行fresh-process恢复及trace验收，不沿用另一个后端的合格记录。现有精确恢复标准和单次更新对比rtol1e-5、atol1e-6、RNG/控制状态完全相同的要求均保留。

新增候选调用`torch.compile(backend="eager", fullgraph=True)`取得Dynamo FX前向图，保留原生autograd，再由`make_graphed_callables`捕获前向与反向。这里的`eager`是PyTorch编译后端名称，实际运行模式必须是`compile_graph`，不是退回普通eager训练。此候选没有AOT重写和Inductor/Triton融合，收益须来自实测Graph replay，不能宣称融合加速。进度界面明确显示“Dynamo原生autograd＋CUDA Graph”。

本次Inductor的跨进程恢复出现模型最大绝对差2.17e-6、Adam最大差1.12e-8，控制和RNG仍相同；AOT有合成数值/RNG不一致，真实校准则在首次replay前抛出RuntimeError。旧报告没有底层错误，不能断言OOM或具体算子。现在把Dynamo前向和懒加载反向先放在Graph捕获外预热，使用`autograd.grad`而不累加参数grad，结束后恢复原RNG；完整更新仍按原快照回滚。新增分阶段错误类型、白名单分类、框架/项目代码位置及构建次数，不上传异常原文、局部变量或输入。编译包装器中的原生OOM仍交给原micro-batch回滚控制器；致命CUDA错误不会重试。合成检查若实际退到eager，立即拒绝该候选，不再做它的真实Graph校准。

Inductor候选关闭epilogue fusion、shape padding及layout optimization，保留FP32、fallback_random和显式Graph。另一候选采用PyTorch内置`aot_eager_default_partitioner`：通过torch.compile/AOTAutograd生成前向和反向图，内部保留原生ATen算子，再交给显式CUDA Graph捕获。它没有Inductor/Triton融合，不应宣称为融合编译提速。选择由当前环境上的正确性、恢复性和完整更新测速决定；均未通过时如实返回失败。

真实更新比较现在分别检查logits、裁剪前梯度、裁剪后梯度、模型、Adam、样本调度及RNG，保存各字段差异计数、最大绝对差和最多32条位置；不会上传张量值或输入。每个新进程先完整预热候选前向/反向/裁剪/Adam/调度，再恢复同一快照后比较稳态更新。Retrieval原生反向分图候选保留两个文档图，其他候选保留一个整前向图；改变执行形状后复核，最多8代构建/模式校验，同形状不重复测速。

Graph捕获保留原生OOM类型。任务要求Graph时，OOM先释放图池，交给既有有限重试控制器缩小micro-batch、恢复同一快照后重新捕获，不因为首次OOM放弃Graph。epoch末不足micro-batch的尾块走原生eager，保留整批主图；整个混合更新仍通过数值校验。按图构建代数缓存校验结果，图实际重建才使旧记录失效，同一代中的整批/尾批交替不会反复耗尽预算。校准报告必须证明Graph实际replay次数大于0，不能仅凭模式名称声称Graph生效。

带完整断点的未完成Torch任务还要通过真实断点校准：隔离worker从不可变migration-source快照恢复模型、Adam、RNG、采样及样本时钟，在当前真实训练样本上比较并测速，随后完整回滚并精确校验；提交样本和优化器更新均为0。校准不访问测试集，不保存正式checkpoint，不触碰原Suite。其配置、checkpoint哈希、GPU及报告纳入证书；失败候选的紧凑报告在重新进入时仍保持原字节。正在等待epoch末验证、没有待更新batch的边界断点会明确拒绝校准，避免为了测速运行验证/测试。

测速仍为一次预热及两个交替测量对，比较包括反向与Adam在内的完整更新；任何测得正收益可保留。这是有界准入测速，不能代替持续吞吐证据。正式worker仍复核真实更新；后续形状失配也必须遵守后端要求。

`--require-graph transformer-retrieval-s42`要求当前Retrieval任务真正启用Graph且真实断点测速有收益。该要求下只尝试三个Graph候选；不能通过时自动回传诊断并停止，不静默切到compile/eager长跑，不提交失败更新。`--require-acceleration JOB_ID`允许compile或Graph。其他任务保留有限回退；S5继续原生JAX。

## 从当前840断点采集

以下命令须在本次实现成功发布后使用。先以原14978da目录准备独立新checkout，共享已有离线环境/.bootstrap，不改变原840执行代码。若原840后来重新启动过训练，须先正常安全暂停并等保存完成；新入口也会拒绝持锁中的训练源。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-14978da5b3cf &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/collect_operator_profile.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpu 0
```

`--configure`复用完整环境或私密文件；只有未配置时才在终端询问原地址/token，token隐藏输入。配置保存到账号私密目录并可供新终端/worktree复用。使用现有凭证，不重新申请、不发到聊天或Git。采集只依赖现有PyTorch2.7.1环境和本地数据，不在线安装profiler或下载公共依赖。原认证快照里的dataset路径必须仍可用。

预期终端显示`Operator-only Suite`、本次目录、上传SHA-256核验及采集状态，随后退出。没有训练进度、不会启动监督器。返回状态为collected才说明有GPU算子归因；unavailable或cuda_attribution_unavailable需要按返回原因修复采集环境。无论何种状态，采集不会签发正式release，后续提速改动仍需原有正确性、恢复和收益验收。

若只上传失败，使用终端本次真实Suite路径传给`experiments.baseline_delivery.return_artifacts --directory ... --experiment operator-profile --file operator-profile.json`重传；不要为了重传再次采集。未完训练后续仍从原840正式训练源迁移，不能将`operator-profile-*`当作训练源。

## 原生双文档并行候选（单任务诊断）

RTX3090回传表明，原生`_efficient_attention_forward/backward`合计占本次算子self CUDA总和88.75%；这已经是融合注意力，Graph本身未带来正收益。单次算子总和不能视为关键路径耗时。固定v2.7.1源码中，严格确定性将efficient attention反向的key分片数限制为1，Flash/cuDNN则要求低精度；本候选不改变这些设置。

`eager_parallel`仅由`scripts/check_parallel_attention.sh`显式选择，不加入原25任务后端轮询。它对Transformer/Retrieval创建两个临时编码器副本，各自使用`torch.compile(backend="eager",fullgraph=True)`及独立前向/反向CUDA Graph池。两路前向使用设备依赖按文档1、文档2执行，保护默认生成器共享的设备seed/offset；两路反向仍使用独立流。每个micro-batch先同步原模型参数与buffer；副本仅持有独立梯度，按文档2加文档1的固定顺序在调用流上合并，交给原模型和Adam。跨流输入、输出和梯度通过`record_stream`保护存储生命周期。原参数、checkpoint键、逻辑batch、注意力算子、FP32、dropout与确定性要求均保留。编译后端eager不代表Inductor融合；并行提交也不保证GPU能有效重叠或产生提速。

该梯度合并顺序是必须验证的执行变化，不能从代码或模拟测试推断与旧执行数值相同。专用入口只做这一个候选的logical32/64/128新进程精确恢复与trace检查，再从原840不可变断点进行真实完整更新数值/RNG检查、一次预热与两个AB/BA测量对、精确回滚。单图仍受原uint32 RNG容量约束，超限/OOM走原有限物理降档；不存在无限重试或待测触发重训。此检查不再重复采集算子或无收益分阶段耗时，不启动正式训练、不签release。

当前成功采集目录为`/root/projects/qkan-mamba-lra-adaptive-a8dc703e9f8f`，从此处准备已发布新版本后执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a8dc703e9f8f &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_parallel_attention.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpu 0
```

原840若已重新训练，先由师弟安全暂停并等待保存；入口会拒绝活跃源锁。仅使用原840训练源，不使用任何失败准备或operator-profile目录。原两个已完成结果和来源记录保留。终端输出每个恢复子进程及真实校准阶段，结束后自动上传`parallel-attention/<本次Suite>/parallel-attention.json`并核验SHA-256。失败也回传紧凑诊断；上传失败保留原报告，按同目录/experiment/file重传即可，不为回传重新计算。只有真实校准报告精确回滚通过后才声明0样本/0更新，合成检查产生的临时更新只在新诊断目录内。

原生通过、正收益和长期持续收益均待师弟实测；任何失败不会静默开启eager长跑。正式迁移仍需后续受审准入，诊断目录不能作为训练源。实现依据为固定官方[CUDA流与反向语义](https://github.com/pytorch/pytorch/blob/v2.7.1/docs/source/notes/cuda.rst)和[独立Graph池/自动求导](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)，源文件哈希记录于本次交付证据。

## 并行候选卡点复测

2026-11-12回传的2b3df03尝试在第一次恢复检查中以KeyboardInterrupt结束，旧进度约36.9小时未更新；没有通过恢复或测速，上传成功只表示报告已回传。旧快照不能定位具体阻塞操作。

专用诊断现为每个恢复/校准子进程设置`--child-timeout 900`秒总上限和`--stall-timeout 300`秒无阶段推进上限。普通训练入口不采用这些短时限。每15秒显示PID、阶段和耗时；独立心跳每5秒发布Python栈位置，心跳刷新本身不重置阶段超时。新增阶段覆盖数据/模型准备、断点恢复、副本创建/同步、编译前向、预热反向、同步、Graph捕获、两文档前反向及trace。

启动前按选定GPU的UUID检查compute进程，有占用或无法核验则停止并回传。不得自动结束历史或未知进程。超时先向本次bootstrap已登记的PID请求一次faulthandler栈，再向本次新建进程组发TERM，10秒后必要时发KILL并核验Linux非僵尸成员已清空。无法确认退出会明确记录，不能开始下一子任务。原始栈和完整日志只留在本次私有诊断目录；自动回传只含有界的代码文件/函数/行号、阶段、时间、PID和退出核验，不含源代码行、局部变量或张量。

从师弟当前2b3df03目录准备独立最新版后复测：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-2b3df03a444d &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_parallel_attention.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpu 0 \
  --child-timeout 900 --stall-timeout 300
```

该版本先定位卡点，不宣称已经修复底层CUDA阻塞或获得提速。它仍保留原完整数值/RNG、精确恢复、实际Graph replay及正收益门槛。阶段标记不查询/同步CUDA，不改变权重或RNG；冷构建阶段立即落盘，热执行标记由后台定期落盘。所有恢复/校准子进程的退出记录合入`parallel-attention.json`，失败也用原认证通道自动回传。未通过时不启动正式训练，原840断点和两项已完成结果保持为来源。

## 300秒超时后的GPU阶段定位

2026-11-12的`parallel-attention-20261112T105040Z-c9265218`首次子进程在最后记录`backward_join`后触发阶段超时。认证报告核验39834字节，SHA256为`cad473cf642253a927773dcb65d1684e536401aaaa34895e232cacf844b01e13`。Python栈实际停在更新末尾的`bool(finite)`，即CPU等待读取CUDA标量；此前异步提交了多轮前反向。最后阶段名不能证明阻塞操作就是join。退出码-9，本次进程已回收，Linux进程组核验为空，自动回传成功。

固定v2.7.1的[CUDAGraph::replay](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAGraph.cpp)与[CUDAGeneratorState::replay_prologue](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)显示不同Graph回放会写默认生成器共享的设备seed/offset。修订为第二文档前向流等待第一文档流完成，保证设备执行顺序；仅按CPU提交顺序无法保证这一点。另依官方CUDA流文档补上跨流张量存储记录。这些是代码风险修正，尚无原生证据证明它们解决了本次GPU阻塞。

新增`--stage-fences`仅对本次受监督的合成恢复检查启用：每个文档的前向、反向及梯度合并后，在写出明确阶段后等待当前CUDA流完成；更新末尾也有等待边界。普通模式不增加这些等待。该诊断会串行化被观察的阶段，不能证明双流并发正确或产生收益；通过后仍须无该参数的原生恢复及真实断点校准。它不执行速度校准、不签加速准入，顶层`passed=false`；成功仅标记`stage_observation_passed=true`。真实训练/校准worker拒绝启用该模式。环境标志在父进程中精确恢复。

师弟从当前a36目录准备新工作目录后，先做这次有限定位：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a36a271fa95b &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_parallel_attention.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpu 0 \
  --child-timeout 900 --stall-timeout 300 --stage-fences
```

结束后同样自动回传`parallel-attention.json`；无需为回传重跑计算。原840训练断点及两项已完成结果仍是唯一来源。阶段观察通过后，去掉`--stage-fences`才会执行原有完整恢复和真实断点AB/BA校准；所有数值、RNG、精确恢复、精确回滚与正收益门槛保留。新版本的原生阻塞修复与提速均待验证。

## 显式隔离每个文档的Graph捕获流

2026-11-13回传`parallel-attention-20261113T005247Z-6a526c1a`表明，a39在关闭`--stage-fences`后仍于首次恢复子进程超时。最后阶段为`gradient_merge`，信号栈位于`fixed_merge`的张量加法；本次子进程已回收，Linux进程组为空。上次逐阶段等待检查的通过只适用于带额外同步的执行，正常并发与速度仍待验证。

源码复查发现：v2.7.1的`make_graphed_callables`调用`torch.cuda.graph`时未传stream，两个独立内存池仍使用同一个默认捕获流。该版本的[cuBLAS工作区缓存](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CublasHandlePool.cpp)以(handle, stream)为键；独立副本和Graph池本身没有隔离这个库工作区。因此，捕获在同一流的两个反向图并发回放，存在共享工作区的风险。这是源码支持的风险判断，尚非本次阻塞的原生根因证明。

修订用公开`CUDAGraph`和`torch.cuda.graph(...,stream=本路流,pool=本路池)`分别捕获每个文档的前反向，并用自动求导桥接静态输出/参数梯度。每路内部前向、反向共用本路池；两文档使用不同捕获流。整数输入无梯度，副本参数梯度保持原固定合并顺序；正常回放不增加主机同步。默认生成器的前向设备顺序约束继续保留。没有启用AMP、TF32或替换注意力，也没有将Dynamo eager标记成Inductor融合。

师弟从现有a39目录准备更新后，直接运行关闭逐阶段等待的检查：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a39f7637f40e &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/check_parallel_attention.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpu 0 \
  --child-timeout 900 --stall-timeout 300
```

本次只检查该修订的精确恢复/trace和真实断点完整更新数值/RNG、速度及精确回滚，结束自动回传。原840断点和两项已完成结果保留，正式训练不在此入口启动。旧a39工作目录仍可用于版本回退；它的未加等待版本已有超时证据，应保留用于审计。新版原生阻塞修复和正收益均待师弟回传。

## 已验证并行Graph路径的正式续跑入口

2026-11-14，`abe8694`的无`--stage-fences`报告已通过精确恢复/trace、原840断点完整更新数值/RNG、实际Graph回放及精确回滚。两组AB/BA更新中位数为普通8.899131秒、候选7.899208秒，更新耗时降低11.2362%，对应吞吐提高12.6585%。[核验记录](../evidence/parallel-attention-native-gain-review-20261114.json)仅支持该断点、设备和候选；持续训练收益仍需观察。

`scripts/migrate_parallel_retrieval.sh`显式启用`--parallel-retrieval-graph`：只有未完成的`transformer-retrieval-s42`选择`compile_graph / eager_parallel`，并要求实际Graph与正收益；其余Torch任务选择普通执行，S5保持原生JAX。不会重复轮询此前无收益的Inductor/AOT后端。两个已完成结果沿用原字节和来源，剩余任务按原25任务队列继续。

师弟从当前abe目录执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-abe8694cab9e &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/migrate_parallel_retrieval.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpus 0
```

新目录重新签发与新版本、配置、原断点和设备绑定的检查记录；不会改写abe的通过报告。入口先执行来源、恢复/trace和迁移检查，并对Retrieval做900秒绝对/300秒阶段停滞保护下的恢复及真实断点校准，自动回传紧凑证据；全部通过后自动进入进度界面并训练。其他任务仍通过完整队列的原有校验。加`--gates-only`则检查后停止，不启动训练。任何Graph无收益/数值不符都停止，不把Retrieval静默降成普通训练。

后续中断后，在终端打印的**新训练实验目录**上用同版本`formal_progress.py --resume`续跑；不要把`parallel-attention-*`诊断目录作为训练源。保留原840目录用于原版本回退；新入口仍沿用完整状态恢复和短期/长期速度显示。

## 进度显示

短期速度是训练心跳时间戳对应的近60秒样本斜率，窗口逐步积累；重复刷新同一心跳保留测量值。长期平均采用本次进程的已提交样本增量除以墙钟耗时，包含等待/保存和已发生的验证，排除旧checkpoint前缀。旧worker无起点信息时标注“本次观察”。当前任务ETA按长期平均估计，不包含后续验证、测试、回传或其他排队任务；启动阶段平均会受编译/校准成本影响。

心跳超过120秒未更新时保留并标记旧速度、暂停ETA估计；显示刷新不触发训练、测速或重启。编译/Graph行显示实际模式、准入编译后端及必须启用Graph的要求，避免将已实现或合成准入误读为真实训练正在加速。

## 验证边界

所有权重、数据和原生Torch/JAX运行均留在师弟服务器；本机只运行标准库、注入模拟、AST、CLI与身份哈希检查，不导入模型框架、不查GPU、不安装依赖、不训练。v5代码交付及模拟通过不等于原生Graph通过收益门槛或持续提速。无AMP、TF32、注意力替换、新依赖或运行时公网fallback。提交的release保持关闭，原生证据只在协作者验收通过后产生。

固定官方API依据：[PyTorch v2.7.1 AOT后端](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/backends/debugging.py)、[CUDA Graph](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)、[Inductor配置](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py)。后端可用性仍由各自已安装的原生环境判断。

随机数容量依据：[CUDAGeneratorImpl](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)、[CUDA注意力](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/attention.cu)、[原生dropout](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Dropout.cu)。源码仅作为固定版本行为依据，实际后端与误差仍由师弟环境验证。
