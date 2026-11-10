# Torch 编译、CUDA Graph 与真实断点校准

2026-11-09 最新自动回传`failure-37b2622a59ef`确认：d808f0a的Graph/Dynamo原生反向已通过合成精确恢复及真实断点数值/RNG检查，真实校准实际回放80次，捕获错误已不再出现。这次失败原因是无测速收益：普通执行8.7803204656秒/更新，Graph8.8010682687秒/更新，慢约0.236%。原断点精确回滚，提交样本/更新为0，准备Suite `runs/adaptive-migrated-20261109T134728Z-225944`没有启动正式训练。Inductor仍有精确恢复差异，AOT仍有数值/RNG差异。原840训练源保持`runs/adaptive-migrated-20261109T045101Z-183941`、Retrieval step12683及两个原9a完成结果。最新边界见`docs/evidence/compiler-calibration-measurement-20261109.json`；不能将能回放Graph称为已有加速。

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

## 从当前840断点交付

交付状态2026-11-10：私密配置/仅检查修复已在本地提交，但授权3080的SSH直连超时，现有代理路线也未连通；GitHub仍只有14978da。以下自动配置命令须等修复提交成功发布后使用，不能把本地完成当作已经可拉取。

2026-11-10截图中的14978da在检查交换配置时退出，尚未建立本次Suite、运行模型或采集阶段耗时；这只证明新终端没有继承三个环境变量，不证明已有凭证失效。下面从该目录获取修复版，先保存/复用私密配置，再从原840 Suite迁移，保留其step12683及两个完成结果。本组命令只运行验收/校准，不启动正式长训练。若后来另行恢复了原840训练，须先安全暂停并等待保存。不要把失败的准备Suite当作训练源。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-14978da5b3cf &&
test -d /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.exchange_config --configure &&
bash scripts/migrate_torch_acceleration.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpus 0 \
  --require-graph transformer-retrieval-s42 \
  --gates-only
```

`--configure`优先复用完整环境或已保存私密文件；缺少时在交互终端询问原服务地址和token，token隐藏输入，已有`~/qkan-private/exchange-ca.crt`自动复用，否则询问已有证书路径。一次配置后默认保存到`~/qkan-private/exchange.json`，Linux文件权限0600，新终端及隔离worktree自动读取。支持`QKAN_EXCHANGE_CONFIG`指定绝对私密文件路径；默认还兼容共享`.bootstrap/exchange.json`、`~/qkan-private/exchange.env`及本worktree`.env`的三个字面赋值，不执行source/eval。完整环境优先；部分环境与文件冲突时停止，避免把token发给另一个服务。文件和凭证不得发到Git或截图。

原840和14978da执行代码保持各自版本；只向被Git忽略的.bootstrap写准备脚本。新版目录共享既有离线环境和源码资产，代码、证书及新Suite独立。迁移认证原policy/certificates/gate-evidence及其哈希，沿用完成结果的原始字节和来源，不重新训练或改称新版结果。模型、Adam、RNG、采样顺序/游标、样本学习率时钟、OOM控制器和迁移历史保留；旧compiler选择归档后重新准入。总计25任务、100epochs和样本预算保持声明值。

迁移包装入口现在真正遵守显式`--gates-only`：即使全部验收通过，也不会启动formal_progress训练监督器。无该参数时仍在通过后自动启动训练。此前包装器总会启动监督器，是入口缺陷；本次修正不改模型、校准策略或收益标准。若Graph正确但仍无收益，两次阶段剖析随failure.json自动回传；若测得正收益，则本版不会进入no_gain_phase_profiles分支。

记下新Suite和工作目录；后续普通同版本续跑只用它的`formal_progress.py --resume`入口，不再次迁移。校准失败的报告自动发往`formal-readiness/failure-*`；正式worker因后端要求停止时，紧凑数值和错误类型报告发往`admitted-entry`。数据、权重、原始训练日志、原始trace和凭证不进入这些诊断。

### 发布网络未恢复时：直接使用现有14978da采集

无需等待代码更新。先在当前终端恢复原来的三项交换配置，再直接调用run_formal_readiness.sh；它的--gates-only已经有效，避开旧迁移包装器启动监督器的问题。该入口自动推出adaptive/performance/dynamic选项，保留原840迁移源。已有配置复用，缺少时输入原地址/token；已有CA路径可通过PROJECT_EXCHANGE_CA_FILE保留，默认复用/root/qkan-private/exchange-ca.crt。配置阶段失败时仍不创建Suite，不进入计算。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-14978da5b3cf
if [[ -z ${PROJECT_EXCHANGE_URL:-} ]]; then
  read -rp '原文件共享服务 HTTPS 地址: ' PROJECT_EXCHANGE_URL
fi
if [[ -z ${PROJECT_EXCHANGE_TOKEN:-} ]]; then
  read -rsp '原项目 token（隐藏输入）: ' PROJECT_EXCHANGE_TOKEN
  printf '\n'
fi
export PROJECT_EXCHANGE_URL PROJECT_EXCHANGE_TOKEN
export PROJECT_EXCHANGE_CA_FILE="${PROJECT_EXCHANGE_CA_FILE:-/root/qkan-private/exchange-ca.crt}"
bash scripts/run_formal_readiness.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpus 0 \
  --torch-acceleration --require-graph transformer-retrieval-s42 --gates-only
```

只需原有凭证，不需要重新申请或把token发到聊天。CA实际位置不同时，在本机终端设置其已有路径。若本轮仍为Graph无收益，阶段耗时将随failure.json自动回传。新私密文件持久化功能在修复发布后启用；临时入口不把凭证写进Git。

## 进度显示

短期速度是训练心跳时间戳对应的近60秒样本斜率，窗口逐步积累；重复刷新同一心跳保留测量值。长期平均采用本次进程的已提交样本增量除以墙钟耗时，包含等待/保存和已发生的验证，排除旧checkpoint前缀。旧worker无起点信息时标注“本次观察”。当前任务ETA按长期平均估计，不包含后续验证、测试、回传或其他排队任务；启动阶段平均会受编译/校准成本影响。

心跳超过120秒未更新时保留并标记旧速度、暂停ETA估计；显示刷新不触发训练、测速或重启。编译/Graph行显示实际模式、准入编译后端及必须启用Graph的要求，避免将已实现或合成准入误读为真实训练正在加速。

## 验证边界

所有权重、数据和原生Torch/JAX运行均留在师弟服务器；本机只运行标准库、注入模拟、AST、CLI与身份哈希检查，不导入模型框架、不查GPU、不安装依赖、不训练。v5代码交付及模拟通过不等于原生Graph通过收益门槛或持续提速。无AMP、TF32、注意力替换、新依赖或运行时公网fallback。提交的release保持关闭，原生证据只在协作者验收通过后产生。

固定官方API依据：[PyTorch v2.7.1 AOT后端](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/backends/debugging.py)、[CUDA Graph](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)、[Inductor配置](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py)。后端可用性仍由各自已安装的原生环境判断。

随机数容量依据：[CUDAGeneratorImpl](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)、[CUDA注意力](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/transformers/cuda/attention.cu)、[原生dropout](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Dropout.cu)。源码仅作为固定版本行为依据，实际后端与误差仍由师弟环境验证。
