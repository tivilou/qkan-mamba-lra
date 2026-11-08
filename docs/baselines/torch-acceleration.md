# Torch 编译、CUDA Graph 与进度显示

当前训练源是9ce7f00d45d2，不再是原9a目录。2026-11-08截图确认该Suite已恢复训练：两个原9a结果保留，Transformer/Retrieval正在继续。该任务的compile_graph和compile先前均未通过跨进程精确恢复，正式准入为eager。代码已实现这两种后端，不代表该任务已获准使用，也没有证据证明底层差异源于embedding算子。

## 先更新显示，无需停止训练

在**另一个终端**执行。只提取观察脚本，不pull或修改当前执行版本；`--watch-only`不会启动、停止、测速或修改训练。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-9ce7f00d45d2 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/formal_progress.py &&
.venv-baselines/bin/python -B .bootstrap/formal_progress.py \
  --watch-only --resume runs/adaptive-migrated-20261108T082557Z-145040 \
  --display inplace --interval 5
```

短期速度用训练心跳的时间戳和已提交样本差计算，保存近60秒窗口；不足60秒时逐步积累。重复刷新同一心跳保留最近测量值，不反复清成“待测”。长期平均用本次观察期间的实际样本增量除以墙钟时间，包含等待、保存和观察到的验证耗时。旧worker缺少进程起始计数，因此明确标注“本次观察”；新版Torch/S5心跳另记录进程起点与恢复断点的起始样本/step，能够显示“本次进程”平均，不把旧断点样本算入新进程吞吐。

短期、长期速度分别占一行，避免窄终端截掉长期平均。当前任务ETA按长期平均计算，不包含将来的验证、测试、回传和排队任务。超过120秒无新训练心跳时，保留并标明最近测量、暂停ETA估计；这不是失败判定，也不触发重新测速。无训练阶段事件的旧worker不能确定是在保存、验证还是编译，界面不作虚假归因。eager的准入恢复失败原因从原acceleration-admission.json读取，避免仅显示含糊的requested。

## 编译恢复修正

执行策略升级为qkan-torch-acceleration.v2。每个新进程、每个新的micro-batch/完整batch形状，在数值对比前先完整预热一次前向、反向、裁剪、优化器和样本调度，再恢复同一个更新前快照。之后比较eager和候选的稳态完整更新，最后再回滚，实际样本只提交一次。只预热forward不能覆盖延迟反向编译。

编译模式缓存已校验形状；Graph只驻留一张图，切换回来需要重新捕获并复核。每个后端最多8次形状校验，Graph仍最多8次构建，超过预算有限回退，避免无限预热或测速。新进程重新预热；原有已选模式/测速结论在同版本断点恢复中保留。正式worker仍只进行每次后端选择的一轮有界吞吐比较（各一次预热、两个交替测量对），正收益保留；形状复核不重复速度benchmark。合成恢复canary不根据短期速度选择后端。

这是对冷编译、换形状生命周期的修正，**尚未证实它就是旧版跨进程差异的原因，或已使Retrieval通过Graph**。确定性算法、FP32、关闭TF32及fallback_random等设置本来已启用，不把再次设置这些选项宣称为根因修复。实际逐位恢复门槛不变，所有候选均失败就阻止训练。

失败诊断增加逐字段是否完全相同、差异元素数、最大绝对差、控制状态/RNG结果和最多32条差异位置，不包含张量值。即使后续eager准入通过，失败候选的compact诊断仍进入acceleration-admission/gate-evidence并随原回传流程上传，便于判断模型、Adam、RNG或调度从哪里开始分歧。

## 采用新版训练代码

显示更新可以直接使用上一条命令。要采用v2编译逻辑，需要在原训练监督器按**一次Ctrl+C**，等待完整保存并返回shell；以下命令会建立独立新执行目录，对剩余任务重新验收后继续。不要在活跃训练时迁移，也不要再从原9a迁移而丢掉9ce中新训练的进度。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-9ce7f00d45d2 &&
test -d runs/adaptive-migrated-20261108T082557Z-145040 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/migrate_torch_acceleration.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-9ce7f00d45d2/runs/adaptive-migrated-20261108T082557Z-145040 \
  --cache /root/qkan-cache --gpus 0
```

迁移支持原9a和当前9ce。9ce来源须通过原策略、证书、gate-evidence及关联文件哈希验证；沿用的原9a完成结果继续验证原迁移谱系并逐字节保留，不重标成9ce或新版结果。剩余任务保留完整模型、Adam、RNG、采样游标、样本学习率时钟、OOM上限和既有迁移历史。只重置旧后端选择元数据以允许新版本重新准入；旧选择写入迁移谱系。先执行零更新原生还原/往返校验，不重启学习率预热。不支持的策略或科学配置变更直接拒绝。

入口默认执行剩余任务完整fresh-process恢复和trace验收，选择compile_graph→compile→eager中实际共同通过的模式，自动启动监督器；已完成结果不重跑。记下入口生成的新Suite路径，以后续跑该新Suite。当前旧Suite继续保留，不在原执行目录覆盖科学代码或证书。

## 实现与验证边界

Inductor绑定原forward，保持模型键及优化器引用。独立Graph捕获前向/反向，优化器留在图外；首次microbatch反向后隔离可复用参数梯度缓冲。禁用隐式cudagraph，最多一张驻留图。图池OOM先退到compile；其余容量OOM进入原batch控制器。失效CUDA上下文立即停止。S5继续原生JAX，旧MEGA环境不升级；无新包、AMP、TF32或注意力替换。依赖和源码沿用本地/认证共享资产，运行时没有公网回退。

本机仅标准库/模拟、AST、身份哈希和CLI检查；不导入Torch/JAX、不查GPU、不装包、不训练。真实编译、显存、完整恢复和持续吞吐由师弟服务器验收。代码交付不代表Graph通过或速度收益已测得。源证据与本轮验证见docs/evidence/compiler-warmup-progress-20261108.json。

API核对采用固定PyTorch v2.7.1官方源码：[CUDA graphs](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)、[Inductor配置](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/config.py)、[确定性与自动调优](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/_inductor/runtime/triton_heuristics.py)。
