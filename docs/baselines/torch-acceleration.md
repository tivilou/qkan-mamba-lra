# Torch 编译、CUDA Graph 与真实断点校准

2026-11-09 截图和自动回传确认：当前执行器是840430a414bc，Suite为`runs/adaptive-migrated-20261109T045101Z-183941`。两个原9a完成结果保留；Transformer/Retrieval恢复step10596后继续。Graph合成恢复失败，pure compile合成恢复通过，但真实训练仍发生NumericalMismatch并退回eager。先前交付没有证明该任务获得编译或Graph收益。详见`docs/evidence/compiler-runtime-observation-20261109.json`。

## v3修正

候选顺序为Graph/Inductor、Graph/AOT原生算子、compile/Inductor、compile/AOT原生算子、eager。每个候选独立进行fresh-process恢复及trace验收，不沿用另一个后端的合格记录。现有精确恢复标准和单次更新对比rtol1e-5、atol1e-6、RNG/控制状态完全相同的要求均保留。

Inductor候选关闭epilogue fusion、shape padding及layout optimization，保留FP32、fallback_random和显式Graph。另一候选采用PyTorch内置`aot_eager_default_partitioner`：通过torch.compile/AOTAutograd生成前向和反向图，内部保留原生ATen算子，再交给显式CUDA Graph捕获。它没有Inductor/Triton融合，不应宣称为融合编译提速。选择由当前环境上的正确性、恢复性和完整更新测速决定；均未通过时如实返回失败。

真实更新比较现在分别检查logits、裁剪前梯度、裁剪后梯度、模型、Adam、样本调度及RNG，保存各字段差异计数、最大绝对差和最多32条位置；不会上传张量值或输入。每个新进程先完整预热候选前向/反向/裁剪/Adam/调度，再恢复同一快照后比较稳态更新。Graph只保留一张图，改变执行形状后复核；最多8次构建/模式校验，同形状不重复测速。

Graph捕获保留原生OOM类型。任务要求Graph时，OOM先释放图池，交给既有有限重试控制器缩小micro-batch、恢复同一快照后重新捕获，不因为首次OOM放弃Graph。epoch末不足micro-batch的尾块走原生eager，保留整批主图；整个混合更新仍通过数值校验。按图构建代数缓存校验结果，图实际重建才使旧记录失效，同一代中的整批/尾批交替不会反复耗尽预算。校准报告必须证明Graph实际replay次数大于0，不能仅凭模式名称声称Graph生效。

带完整断点的未完成Torch任务还要通过真实断点校准：隔离worker从不可变migration-source快照恢复模型、Adam、RNG、采样及样本时钟，在当前真实训练样本上比较并测速，随后完整回滚并精确校验；提交样本和优化器更新均为0。校准不访问测试集，不保存正式checkpoint，不触碰原Suite。其配置、checkpoint哈希、GPU及报告纳入证书；失败候选的紧凑报告在重新进入时仍保持原字节。正在等待epoch末验证、没有待更新batch的边界断点会明确拒绝校准，避免为了测速运行验证/测试。

测速仍为一次预热及两个交替测量对，比较包括反向与Adam在内的完整更新；任何测得正收益可保留。这是有界准入测速，不能代替持续吞吐证据。正式worker仍复核真实更新；后续形状失配也必须遵守后端要求。

`--require-graph transformer-retrieval-s42`要求当前Retrieval任务真正启用Graph且真实断点测速有收益。该要求下只尝试两个Graph候选；不能通过时自动回传诊断并停止，不静默切到compile/eager长跑，不提交失败更新。`--require-acceleration JOB_ID`允许compile或Graph。其他任务保留有限回退；S5继续原生JAX。

## 从当前840断点交付

在当前训练监督器按一次Ctrl+C，等待保存完成并返回shell后执行下面这一组。它建立独立新版工作目录，迁移当前840的最新完整断点及两个完成结果，执行全部剩余任务验收和真实断点校准；通过后自动启动训练，不需要另跑启动脚本。不要从历史9ce或原9a Suite再次迁移，否则会丢掉840中新训练的进度。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-840430a414bc &&
test -d runs/adaptive-migrated-20261109T045101Z-183941 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/migrate_torch_acceleration.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941 \
  --cache /root/qkan-cache --gpus 0 \
  --require-graph transformer-retrieval-s42
```

原执行目录保持840版本；只向被Git忽略的.bootstrap写准备脚本。新版目录共享既有离线环境和源码资产，代码、证书及新Suite独立。迁移认证原policy/certificates/gate-evidence及其哈希，沿用完成结果的原始字节和来源，不重新训练或改称新版结果。模型、Adam、RNG、采样顺序/游标、样本学习率时钟、OOM控制器和迁移历史保留；旧compiler选择归档后重新准入。总计25任务、100epochs和样本预算保持声明值。

记下新Suite和工作目录；后续普通同版本续跑只用它的`formal_progress.py --resume`入口，不再次迁移。校准失败的报告自动发往`formal-readiness/failure-*`；正式worker因后端要求停止时，紧凑数值和错误类型报告发往`admitted-entry`。数据、权重、原始训练日志、原始trace和凭证不进入这些诊断。

## 进度显示

短期速度是训练心跳时间戳对应的近60秒样本斜率，窗口逐步积累；重复刷新同一心跳保留测量值。长期平均采用本次进程的已提交样本增量除以墙钟耗时，包含等待/保存和已发生的验证，排除旧checkpoint前缀。旧worker无起点信息时标注“本次观察”。当前任务ETA按长期平均估计，不包含后续验证、测试、回传或其他排队任务；启动阶段平均会受编译/校准成本影响。

心跳超过120秒未更新时保留并标记旧速度、暂停ETA估计；显示刷新不触发训练、测速或重启。编译/Graph行显示实际模式、准入编译后端及必须启用Graph的要求，避免将已实现或合成准入误读为真实训练正在加速。

## 验证边界

所有权重、数据和原生Torch/JAX运行均留在师弟服务器；本机只运行标准库、注入模拟、AST、CLI与身份哈希检查，不导入模型框架、不查GPU、不安装依赖、不训练。v3代码交付及模拟通过不等于原生Graph通过或持续提速。无AMP、TF32、注意力替换、新依赖或运行时公网fallback。提交的release保持关闭，原生证据只在协作者验收通过后产生。

固定官方API依据：[PyTorch v2.7.1 AOT后端](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_dynamo/backends/debugging.py)、[CUDA Graph](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)、[Inductor配置](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py)。后端可用性仍由各自已安装的原生环境判断。
