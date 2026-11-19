# 解析K梯度候选的正式续跑入口

> 日常启动、续跑、看板和回传现统一使用[实验入口](experiment.md)。当前已存在的速度优化版Suite使用该页首节一体化续跑命令；本页保留原控制协议及历史独立观察命令。

本入口交付完整25任务队列的续跑控制代码。实际worker的零更新迁移、科学状态2步／1＋1步精确恢复和真实固定案例轨迹已经通过[认证审阅](../evidence/key-bias-worker-native-review-20261117.json)。数值、64步稳定性及现有执行器比较沿用既有证据。此次新增的是独立长期启动能力与队列分派；18份受审阅执行文件及验证器均保持原字节，常规worker CLI和历史有限准入入口不变。新入口控制测试由本机QT2标准库执行，实际迁移、整份验证集重新评分和正式训练由合作者执行。

## 执行范围

- 从暂停的b07原Suite读取最新兼容完整checkpoint，复制独立快照和结果；不使用旧诊断Suite的更新后状态，不回退到旧step以配合检查。
- 保留Transformer/ListOps和Text两个已完成结果及其9a147cd来源、文件哈希。新manifest将它们标为保留结果，既不重新训练，也不把新入口版本写成旧结果的执行版本。
- 只有未完成Transformer/Retrieval启用`bf16-key-bias-analytic-zero-v1`：BF16注意力、FP32主参数／投影／残差／分类头／损失和Adam，四段K偏置梯度解析置零，eager物理16／逻辑128，不组合Graph。
- 其余22项未完成任务使用原b07执行器、原配置和原运行时证书；MEGA／S5使用已有独立解释器，S5保持原生JAX。新控制器不把旧证书冒用于新候选，也不更改其余模型的执行文件。
- 单卡顺序调度；原25任务顺序、seed42、100轮完整数据预算沿用。历史训练前缀与候选续训分别记录，不把改变梯度定义后的Retrieval标为独立FP32基线。

冻结控制协议见[正式续跑计划](attention-key-bias-formal-plan.json)。GPU必须是已经审阅的同一张RTX3090，且启动时空闲。入口取得原Suite、监督器和所有原worker的内核锁；忙卡或被占用源会退出，保留已有工作。

## 启动前的零更新准备

新Suite首先保存所有任务的最新源快照、原准入身份和两个完成结果。候选配置只改变已声明的策略和新执行版本来源，完整模型、Adam矩、step、样本时钟、排列、历史和梯度规则由原始源及显式转换记录绑定。

随后两个受监督子进程执行：

1. 复用已审阅实际worker的零更新迁移／保存／新读比较，生成本次正式源副本。这不是重跑2＋1＋1更新或性能试验。
2. 用原checkpoint中保存的历史最佳权重，在完整验证集上按候选BF16评估路线重算一次验证分数，核验实际四层／双文档BF16调用覆盖。保存原FP32分数、原best epoch及权重；重算后的分数用于与后续候选epoch按“验证准确率最大、最早同分”比较。历史记录不改写、测试集不访问。除`best_accuracy`及新增迁移说明外，全部训练字段哈希保持精确一致，并验证完整状态保存／重读。

这一步避免旧FP32最佳分数与新BF16评估值混用，以及最终重新评估最佳权重时的来源混淆。没有历史最佳权重时保留未选中状态，不凭空产生权重或分数。

每个准备子进程总计900秒、停滞300秒，0训练更新。准备完成后，入口必须成功自动回传认证`ready`回执，才启动长期训练。任一检查或必要回传失败都保留本地状态并退出。长期训练按原科学预算运行，不套用900秒诊断上限。

## 首次正式启动

在已暂停的原执行目录运行下面整段。准备新代码的输出会包含通用`gates-only`提示；本段随后直接调用新的正式入口。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/continue_key_bias_training.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0/runs/adaptive-migrated-20261114T040717Z-233292 \
  --cache /root/qkan-cache --gpu 0 \
  --worker-from /root/projects/qkan-mamba-lra-adaptive-b23e1da57f99/runs/key-bias-worker-20261117T113204Z-d9dd1cca
```

入口先打印`Formal continuation Suite: .../runs/key-bias-formal-...`，请保留这个新Suite路径。正常准备通过后同一命令继续正式训练，不再等待手工更换命令。若只想执行零更新准备，可加`--prepare-only`；默认即继续长期训练。

## 进度看板与已运行旧版本

正式父进程复用项目原有`scripts/formal_progress.py`美化看板：TTY原地刷新、退出恢复光标；重定向时输出可读快照。机器状态仍写入`heartbeat.json`，终端不再逐条打印完整JSON。界面列出任务汇总、整卡GPU占用、样本／Epoch／实际Step、loss、近60秒平滑速度、本次进程平均速度和当前任务ETA；ETA不代表整个25任务队列的预计完成时间。

`a356914`首次正式入口直接打印JSON，缺少看板所需的父进程`started`／`pid`字段。该版本已经运行时，保持原执行checkout和原进程，另开终端运行下面整段即可恢复原美化看板。新观察脚本从worker心跳中的`run_started_at`读取真实启动时间，兼容旧状态；不更换已绑定的正式执行文件，不触发代码／配置版本迁移。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a3569140d43b &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/key-bias-progress-display.py &&
.venv-baselines/bin/python -B .bootstrap/key-bias-progress-display.py \
  --root "$PWD" --latest-formal --watch-only --display inplace
```

此命令只读本checkout的Suite／状态／心跳，附加有界`nvidia-smi`整卡查询；优先选择唯一标为运行中的正式Suite，没有运行项时选择最新正式Suite。若多项同时标为运行，需用`--resume`显式指定观察的Suite。`--once`可打印一次快照后退出。观察器的Ctrl-C只关闭看板，当前训练继续；原训练窗口仍显示旧JSON是旧进程保留原代码的正常结果。不要对正在运行的绑定checkout执行`git pull`，也不要为切换显示重跑首次迁移或新启动训练。

## 暂停与恢复

保持在**创建该Suite的同一个固定checkout**中。正常Ctrl-C／SIGTERM／SIGHUP沿用原worker的更新边界保存与90秒安全暂停宽限；父进程随后清理自己的进程组，记录退出状态并回传。不得通过重新执行首次迁移命令恢复已有进度。

```bash
bash scripts/continue_key_bias_training.sh --resume /完整路径/runs/key-bias-formal-实际标识 --gpu 0
```

恢复使用最新完整检查点，包括全部参数、优化器、梯度、随机状态、样本排列、cursor、时钟、累计指标和最佳选择状态。完成项先校验后跳过；已经开始的任务丢失完整checkpoint时退出，拒绝退回全新训练。原源、受审阅代码、配置、GPU或入口绑定不匹配也退出。原源仍须保持暂停；若已有新进展，保留两条来源，不将较新进度覆盖到现有分支。

## 自动回传与仅重传

准备、每项完成、暂停和失败都会向`qkan-mamba/key-bias-formal/<Suite--event>/`回传独立且带SHA256回执的紧凑`formal-status.json`。其中包含运行来源、执行绑定、必要恢复控制元数据、指标及代码位置诊断。原始输入、日志、案例完整值及凭据留在本机。训练过程中沿用既有性能profile自动回传；所有25项完成后沿用原`baseline-suite`白名单，上传最佳权重、完整最新状态、指标和紧凑轨迹验证。

若只发生上传失败，使用同一checkout和同一Suite执行：

```bash
bash scripts/continue_key_bias_training.sh --resume /完整路径/runs/key-bias-formal-实际标识 --upload-only
```

此模式只重传已有事件／最终交付，不查询GPU或启动训练。首次快照尚未完成的故障会单独保留`early-failure/failure.json`及其重传清单。私密认证配置沿用账户既有配置，无需重配或写入Git。

## 结果与回退边界

正式训练仍按最佳验证权重评估测试集并生成实际`sample-trace.v2`及语义／head重放。准备阶段只证明迁移和评估衔接，完整收敛、最终模型案例、测试效用和整个队列耗时均等待真实结果。

原FP32源始终保留。需要回退时先安全暂停新Suite，再显式使用原FP32续跑流程；不把候选训练后的权重、Adam或样本进度合并回原源，不将其重标为FP32基线。开发补丁的回滚仅作用于交付文件副本，不删除任何运行Suite。
