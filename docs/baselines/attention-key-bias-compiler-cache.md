# Graph32重编译缓存清理与单项重试

## 已确认原因

[认证原生审阅](../evidence/key-bias-compiler-cache-native-review-20261117.json)对应664b912执行的Suite `key-bias-stability-20261117T020430Z-f2e28cdd`：报告275744字节，SHA256 `2c060b8fccf1c099c99d92bb47a4b1a3b07656a55e3eb84bbc7cf48496702659`。23项代码/计划绑定通过；六项旧结果及旧子进程记录精确复用，本轮只有一个partition32子进程。退出前保留7/8条计时，原完整状态回滚、原源文件不变和进程组清理均通过。

内部错误为`FailOnRecompileLimitHit`，类别`recompile_budget`，阶段`parallel_graph_capture`，第三个捕获代际触发；外层因此拒绝继续Graph并抛出AccelerationRequired。前两个代际的同微批完整更新验证通过，三次Graph32计时及四次BF16计时各自输出和完整状态哈希精确。少最后一条Graph计时，故不计算完整ABBAx2收益，不将部分结果作为准入证据。

代码中的Graph释放会删除副本和可调用对象，未清理Dynamo缓存；多次重建两文档/两分段的Half.forward会在共享代码对象上积累特化。这是与错误位置一致的机制解释；报告没有guard原文/计数，具体特化谓词尚未确认。没有OOM或数值不一致的返回证据。

## 修复边界

只改隔离诊断入口：每次切回partition32 incumbent，先释放旧Graph、恢复原完整起点、同步，再调用`torch.compiler.reset()`，核对完整状态哈希/RNG未变，再进行原捕获/预热/同微批eager完整验证。三个清理位置为计时index0/3/7之前；计时前后仍要求同一驻留捕获和验证代际。记录每次清理的API、位置及状态不变证明，清理不计入更新时间。

[PyTorch v2.7.1官方源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/compiler/__init__.py)明确该API清理编译缓存并建议在无关编译之间使用；其调用Dynamo reset。这里每次已销毁的副本捕获为独立准备阶段；仍只驻留一条执行路线。不增大重编译/捕获预算、不改变原micro32/logical128/dropout/Adam/门限，也不修改正式执行器。API失败或训练/RNG状态改变仍停止并精确回滚，保留诊断。

## 执行

原训练保持暂停、GPU0空闲，在tmux执行以下命令。仍复用最早审阅过的5d7ec8f六项前缀，不使用664b912失败子项拼补计时：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_CACHE_FIX_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_CACHE_FIX_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_stability \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --screen-from /root/projects/qkan-mamba-lra-adaptive-91b8df662c39/runs/key-bias-20261116T090926Z-60f6f2a2 \
  --benchmark-from /root/projects/qkan-mamba-lra-adaptive-b3a337d4267f/runs/key-bias-benchmark-20261116T124533Z-26d47029 \
  --reuse-from /root/projects/qkan-mamba-lra-adaptive-5d7ec8f0cb9f/runs/key-bias-stability-20261117T003301Z-fd47ab92 \
  --child-timeout 900 --stall-timeout 300
```

所有原目录、请求/报告和账户私密交换配置保留。重试仍只有一个partition32子进程，不重复64步、32+32或Graph16；同一原起点/ABBAx2计时从头完成八条，失败旧记录只作为诊断证据。保持900秒子进程/300秒阶段上限，自动回传key-bias-stability.json及验证回执。若源状态推进、GPU忙或绑定不符则停止并保留现有进度。正式样本/更新为0，准入继续关闭。

## 验证与回滚

475项标准库/模拟控制：472通过、3跳过；新增4项覆盖第三次捕获缓存耗尽的旧行为、释放后清理使完整ABBAx2通过、清理在计时外/完整状态不变，以及清理异常时材料保留和精确回滚。既有诊断/前缀复用控制继续通过。该验证不含本机模型/GPU或原生框架导入，实际Graph32修复待合作者回传确认。

六个交付文件的原件、候选、补丁、15条逐字命令与回滚位于`.runtime/key-bias-compiler-cache-20261117/`。独立副本验证旧行为为7条计时/编译预算拒绝，修复行为为8条计时/完整比较通过，两者状态回滚精确；补丁、实际回滚、再应用均验证。八个正式核心及六个复用源/计划文件保持原哈希。`rollback.py <目标副本绝对路径>`仅处理此包六个交付文件。
