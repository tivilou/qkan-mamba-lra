# Torch 编译与 CUDA Graph 可选执行路径

开启 `--torch-acceleration`（或使用下方迁移入口）时，同时启用之前的FP32预取/锁页复用/性能回传。原始9a147cd实验仍是本次迁移来源，用户确认尚未更新。已完成结果原样保留；未完成任务从完整断点继续，不重新预热样本学习率，不改变逻辑32/64/128和物理档位策略。

## 实现范围

- 对模型原有forward绑定 `torch.compile(backend='inductor', fullgraph=True, dynamic=False)`；不替换原模型对象，模型state_dict键与优化器参数引用保持原样。禁用Inductor隐式cudagraph，随机算子使用fallback_random，以便单独管理图捕获及默认RNG状态。无AMP/TF32或新注意力实现。
- 使用 `torch.cuda.make_graphed_callables` 捕获一个适配模块的前向/反向，3次预热；优化器、裁剪、样本学习率、采样游标和checkpoint均在图外。动态输入形状变化时释放旧图，最多同时保留一张形状图，每进程最多尝试8次构建，避免无界图池。图池循环引用主动回收。
- 每个完整更新中，首次图反向后克隆已有参数梯度，防止Autograd累积持有Graph的可复用静态梯度缓冲区。未使用参数的None梯度不被改成零梯度。每个microbatch均完成反向后才复用图输出。
- 每次新进程首次使用加速后端，先以相同模型/优化器/scheduler/RNG和同一批样本做eager与候选完整更新校验。参数、梯度、优化器浮点/复数状态采用rtol1e-5/atol1e-6；控制值、整数和RNG严格相同。所有试更新随后恢复，不推进样本/Adam计数。校验失败进入下一回退路径，不放宽阈值。后续原生fresh-process恢复canary仍沿用原要求，Transformer不改成近似恢复。
- 正式worker另执行有界测速：各1次预热、2个交替测量对，包含前向/反向/裁剪/优化器/调度和末尾同步；每次都恢复完整更新前状态。候选中位耗时只要更低就保留，否则选择eager。每个后端选择只测一次；恢复后的进程保留已选后端/测速结论，但重新构建运行时缓存并重新校验数值。此短测速不保证全程最优；未来形状/显存/负载变化可能改变收益。
- 有限回退顺序为compile_graph → compile → eager。捕获失败、编译器错误、数值不符均恢复相同更新前状态，再重试同一批。图池OOM先释放图，尝试compile；compile/eager的容量OOM交给原有物理batch控制器。非后端模型异常不被吞掉；illegal memory access/device-side assert等失效CUDA上下文明确终止。每种候选最多一轮，避免无限重试。
- S5继续原生JAX。MEGA复用其已固定环境；没有torch.compile时记录回退eager，不升级旧Torch，不安装依赖。Torch2.7.1/triton3.3.1沿用现有基础环境。Inductor/Triton缓存仅写入各任务compile-cache目录，关闭远程缓存，不引入公网下载。

原生合成canary不根据时间选择后端，强制请求的策略并记录是否回退，检查完整状态的跨进程恢复。canary回退后通过只证明该实际执行路径可恢复，不能宣称Graph已通过。编译/图捕获/丢弃的试更新耗时属于真实入口开销；正式样本只提交一次。前16次性能报告的update_wall含初始化/校验/测速，试更新阶段的CUDA细分计时关闭，不能把初始化wall时间当作稳态samples/s。

## 师弟一次更新入口

在原训练监督入口按一次Ctrl+C，等待保存完毕并返回shell，再执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-9a147cd70ced &&
test -d runs/adaptive-migrated-20261106T111646Z-90465 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/migrate_torch_acceleration.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-9a147cd70ced/runs/adaptive-migrated-20261106T111646Z-90465 \
  --cache /root/qkan-cache --gpus 0
```

源目录保持原版本。入口创建独立新版checkout/Suite，保留暂停时所有已完成结果，验证未完成任务的源码机制、恢复/trace以及断点原生零更新还原，然后自动启动监督器。只需这一个入口，无需再单独运行前两个迁移脚本。记下新Suite路径；以后在同一新checkout运行 `scripts/formal_progress.py --resume 新Suite --display inplace --interval 5`。

首次编译、图捕获和有界校验/测速可能较慢，界面显示准备状态而非虚报训练进度；首次编译耗时与机器环境有关，代码没有未经验证的固定分钟数承诺。运行期间显示有效的compile_graph/compile/eager及选择原因。`torch-acceleration.json`记录运行时图构建次数和驻留情况，checkpoint仅保存可序列化的策略选择，不保存CUDA指针或编译对象。前16次性能报告附后端状态并提前自动回传；最终套件回传也包含诊断文件和summary的后端信息。

## 验证边界

2026-11-08恢复准入修复：原250f2fd在Transformer/Retrieval跨进程张量完全一致检查中失败。旧回传未包含具体状态路径/后端，不能断言底层差异来自哪一算子。现按任务在所有选定GPU上完整检查compile_graph；若出现恢复状态差异，则用独立目录从头检查compile，再失败则检查eager。worker内部降到更低模式时也从头复核该实际模式。只有共同通过的模式才写入正式config，作为该任务后端上限；所有模式均失败则阻止启动，不放宽恢复阈值。数据/trace错误或失效CUDA上下文不触发这层回退。acceleration-admission.json记录选择及失败轮次，绑定证书并回传；失败报告新增状态字段路径和两边模式。

若已在250f2fd新目录中报错，保留失败Suite。从原9a目录重新执行上方一次更新指令，仍以原9a完整断点为来源，创建新的修复版checkout和Suite；不要对旧失败Suite执行跨版本resume，也无需修改/删除旧结果。新模式及实际加速收益仍需服务器验证。

本机仅标准库/模拟控制流、AST、hash/provenance和CLI检查，不导入Torch/JAX、不查GPU、不装包、不训练。Torch实际编译、CUDA Graph的前向/梯度与RNG行为、显存占用、跨进程恢复及吞吐必须由师弟服务器自动验收；不承诺一定启用Graph、一定提速或保持与eager逐位相同的未来轨迹。已完成的旧版结果不重新标成新版结果。

API设计核对了固定PyTorch v2.7.1源码：[make_graphed_callables](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)、[Inductor配置](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py)。
