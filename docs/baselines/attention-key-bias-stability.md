# 解析K梯度：64步稳定性与现有执行器比较

前置证据为[认证完整更新实测](../evidence/key-bias-benchmark-native-review-20261116.json)及其原数值筛查、真实配对轨迹和精确恢复。本轮绑定原报告与请求、当前代码/计划、原b07 step14268数据与checkpoint；不重复已完成的四步benchmark。

## 冻结协议

1. 原FP32、FP32解析、BF16解析各64次完整更新，三个独立进程，各自从原完整状态起步；micro16/logical128、原dropout0.1、原AdamW旧矩/step/权重衰减与样本调度。取同一原permutation游标后8192行；开始前必须在当前epoch剩余足够行数。若不足则停止，保留原状态，不重选样本或跨epoch改规则。
2. 逐更新记录CE、正确数、行哈希及注意力调用/dtype摘要，沿用真实运行时的有限loss、有限梯度裁剪和有限主参数检查。将64步固定分为8个连续块，每块8步：两个解析路线相对原FP32的块均值CE超出量均须<=max(0.10,0.25×原FP32块均值)。这是冻结的恶化停止门，不把不同dropout内核视为逐元素同掩码，也不把64步当成完整收敛证据。
3. BF16解析再运行32步、保存有策略/请求/源/原起点绑定的私有状态，新进程加载后再跑32步。与连续64步的全部更新记录、完整状态与分量哈希必须精确一致；成功后仅删除该诊断中点文件和对应回执。中点包含原模型、梯度、Adam、调度、游标/epoch统计、排列与RNG；版本化梯度策略由请求和封装绑定。失败时保留排查材料，诊断状态不用于正式续训。
4. 上述通过后，在两个独立子进程中分别比较已交付执行器：FP32 eager_parallel Graph16、FP32 eager_partitioned Graph32，对BF16解析 eager16。都使用相同原完整起点、相同第一批128行和原dropout，数据选批/H2D、实际损失、反向、裁剪、Adam及样本调度在计时内。Graph32与BF16微批16属于执行方案比较，不声明跨微批或跨策略逐位轨迹相同。
5. 每项比较按incumbent/candidate/candidate/incumbent重复两遍，各4次完整更新计时。每次恢复同一完整状态；每个Graph捕获代际先使用原Executor进行同微批eager对照，完整logits/梯度/增量/状态按rtol1e-5、atol1e-6核验，RNG和控制量精确。捕获/准备/验证、恢复与计时后状态哈希均在计时外。只保留一条驻留执行路线，切换后重建并重新验证；计时前后generation/check计数必须相同且捕获驻留，拒绝计时内首次捕获或降级。
6. 两项各自成功门为中位吞吐收益>0且4个邻接匹配对均>0，同路线4次输出和完整状态哈希精确相同。单项速度拒绝仍完成另一项独立比较并保留两项结果；原生错误则停止后续计算。无收益、损失越限、非有限、恢复不一致、捕获/验证变化均保持正式关闭，不改门限。

七个受监督子进程依次为fp32、fp32_analytic、bf16_analytic、interrupt、resume、graph16、partition32。每个900秒总上限、300秒不变阶段上限，无自动延长、改微批、OOM降级或重跑。三路线192步加BF16中点对照64步，共256次前缀临时更新；执行器比较另有16次计时更新及有界捕获/验证/预热。按前次9.029/3.805秒估计前缀部分约28分钟，额外比较耗时受捕获影响，不能承诺总时长或把它外推25任务ETA。

## 执行命令

先按既有流程暂停原训练并确认GPU0空闲，在tmux执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_PILOT_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_PILOT_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_stability \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --screen-from /root/projects/qkan-mamba-lra-adaptive-91b8df662c39/runs/key-bias-20261116T090926Z-60f6f2a2 \
  --benchmark-from /root/projects/qkan-mamba-lra-adaptive-b3a337d4267f/runs/key-bias-benchmark-20261116T124533Z-26d47029 \
  --child-timeout 900 --stall-timeout 300
```

保留screen及其归因目录、benchmark及其引用的33ee6dd恢复/轨迹目录，复用原本地环境、缓存和账户私密交换配置。原源状态若已推进且不符已审阅14268，停止并保留新进度，不回退正式训练迁就诊断。GPU忙时不终止已有进程，父进程持有原supervisor/suite/worker锁，监督器仅清理自己创建的进程组。

## 输出与回滚

输出`runs/key-bias-stability-<时间>-<ID>/key-bias-stability.json`，自动回传认证交换`qkan-mamba/key-bias-stability/<Suite>/`并核对SHA256回执。报告包括来源与实现绑定、七个子进程退出/清理、全部训练标量与样本哈希、稳定性门、中点恢复、两项计时与驻留验证证据。失败仍回传部分结果及紧凑代码位置，上传失败保留pending manifest。

仅重试上传：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory /absolute/path/to/runs/key-bias-stability-RUN_ID \
  --experiment key-bias-stability --file key-bias-stability.json
```

全部操作在finally恢复原完整状态，移除解析hook/实例策略，释放Graph池，并验证原checkpoint和元数据哈希不变。数据、权重、私有中点和完整轨迹不上传；提交的正式样本/更新为0，不访问验证/测试，不启动正式队列。即使通过，仍须审阅并绑定最终正式代码的恢复、真实轨迹与迁移准入。新[插件计划](attention-key-bias-stability-plan.json)保持approved/native pending；此前真实fixture证明复用能力，未证明本轮尚未执行的Graph交互与64步结果。旧已执行计划与报告保持原样。
