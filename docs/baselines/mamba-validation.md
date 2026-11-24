# Mamba BF16 数值与恢复短验证

三条路线误差量化已完成，见 [实测审阅](../evidence/mamba-noise-native-review-20261123.json)。当前全模型BF16 autocast相对FP32的更新量差异约22.86%～24.49%，明显大于本次各路线自身重复误差；eager与Graph均如此。这版候选未准入，不重复下列已完成诊断。后续若继续BF16，应缩小低精度范围，作为新的数值策略验证；原FP32正式续跑能力保留。

## 当前执行：FP32 融合正式入口＋原美化进度

首次正式入口 `75459d4` 的Suite `mamba-fused-formal-20261124T051405Z-56444863` 在监督器检查阶段退出：启动向量遗漏其要求的唯一 `--output`。认证失败报告574字节/SHA `050b643c57e1e87a90bbee9654886a058601709e2fec87f7a7c68987d242fc9c`；预检报告280字节/SHA `79faa94b9f5bacd55d7dc3e1982b89433d8c628a5f38c9e6551fe53ef8618b45`，子进程记录为空、结果为空、正式更新0。报告中的disposable_updates=4是计划预算，本次实际0。修复仅补齐启动向量的输出目录，并由worker核对它与Suite/job/mode的绑定；不改变训练/恢复策略。旧失败Suite保留证据，不在原固定revision上改文件；下面命令准备修复工作树，并从未变的原训练Suite重新准备。

`start --profile mamba-fused` 创建独立Suite，保留原25任务名单、七个完成结果以及原配置/检查点；仅三个未完成Mamba任务使用已测过的FP32融合训练路径。原worker及其相对导入从原固定checkout加载，进程内包装模型前向、检查点执行标记与心跳；原文件和虚拟环境不改。验证/测试保持普通FP32路径，micro/logical自适应与原样本时钟继续使用，执行器固定eager（无BF16/Graph）。保持科学配置不变，新增融合执行谱系与实现SHA绑定。

同一命令先自动运行实际worker连续2步、单步保存、新进程恢复1步，共4次临时更新；复制完整状态精确核验，恢复起点精确，结束状态按既有rtol1e-5/atol1e-6比较全部科学字段，RNG/样本/时钟/控制项精确。复用FP32非逐位确定性的已有证据，不要求浮点哈希完全相同。检查通过、临时检查点清理并自动回传后，自动从原24981检查点继续正式训练，临时更新不进入正式状态。失败则停在紧凑报告，不开启训练、不自动回退。服务器实际恢复验证待执行；本地仅标准库/模拟验证。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-75459d42205c &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_FUSED_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_FUSED_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_FUSED_REV:0:12}" &&
cd "$MAMBA_FUSED_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py start \
  --profile mamba-fused \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0
```

新Suite路径显示在界面和退出信息中。之后对该Suite使用同一固定checkout的 `scripts/experiment.py resume --suite <新Suite绝对路径>`，继续同屏显示；不要重复start创建第二份。Ctrl-C沿用原安全暂停。源Suite由锁和文件SHA保护，保持暂停；七份完成结果不重新计算。回退可在新入口退出、无新进程后使用原固定checkout的原Suite续跑，原进度保留；融合后新增进度留在新Suite，不声称可以回写旧执行器。

预检自动回传 `qkan-mamba/mamba-fused-worker/<新Suite>--preflight/preflight.json`；正式计算全部完成时回传执行谱系快照与原白名单结果。代码交付未代替服务器上的实际保存/恢复验证，也未证明image/pathfinder的具体加速幅度。MEGA/S5/S4D仍走原执行器。

## 已完成实测：FP32 融合路径吞吐提升17.29%，正式接入待完成

2026-11-24 回传 `mamba-convolution-25d12e7bcca1` 已核验，见 [实测审阅](../evidence/mamba-convolution-native-review-20261124.json)。四个进程退出0且完成清理；FP32完整更新中位耗时 **0.413793646秒 → 0.352782322秒**，吞吐 **+17.2943%**、耗时约 **-14.7444%**，四组成对测速均更快。候选每个两步前缀记录96次实际融合调用，保持FP32、micro8/logical32、原Adam与dropout，未开启BF16或Graph。

相对普通FP32，更新量相对L2 **4.60e-6～5.15e-6**，原始梯度 **7.94e-7～1.17e-6**；同路线重复的更新量差异 **2.36e-6～3.22e-6**。跨路径差异很小但略高于本次重复差异，两步观测不等于收敛或统计误差上界。科学初态、样本/RNG/时钟等控制一致；完整初态哈希含不同执行策略元数据，不要求跨策略哈希相同。

原生依赖导入、兼容包装与融合前后向已在实验机执行成功，源检查点未变，临时二进制已清理，正式提交0更新。原训练仍暂停。下一步只做原worker的FP32融合接入与有限保存/恢复验证，再交付同终端进度界面的续跑入口；不重复本次测速、不扩大到全量检查。以下为已执行的诊断命令与交付细节，保留备查。

已准备 `causal-conv1d 1.5.2` 的官方 Linux x86_64 / CPython3.11 / Torch2.7 / CUDA12 / CXX11 ABI TRUE wheel，133024393字节。下载SHA256 `fa25189970ad71ece008551f11e4b7ce1a6d702ea0609617624d205d2c4408e5` 与 GitHub 官方发布摘要相同；两个离线分块已上传并核验收据，清单 `experiments/baseline_delivery/mamba-conv-wheel.json`。实验机仅从现有认证交换服务下载，不访问外网、不编译、不执行 pip。

`--fused-fp32` 在新诊断工作树的 `.bootstrap` 解包，只有候选子进程将该目录加入 Python 搜索路径。原 `.venv-baselines` 不变。原 Mamba2.2.4 期待旧式CUDA返回值接口；1.5.2改为输出缓冲区接口，其官方 `cpp_functions.py` 正好提供旧式7参数前向／10参数反向的包装。诊断仅将 Mamba 的 CUDA 调用绑定到这两个官方包装，不修改上游源码、二进制或模型参数。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-63676e9581d7 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_CONV_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_CONV_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_CONV_REV:0:12}" &&
cd "$MAMBA_CONV_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --fused-fp32 \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0
```

- 复用现有四进程量化入口，FP32普通路径A/B与FP32融合路径A/B各2步，最后一个进程2次预热＋ABBA×2完整更新测速，共18次一次性更新。保持micro8/logical32/dropout/Adam/样本时钟。没有BF16、Graph、新恢复试验或正式更新。
- 同一测速进程的原路径通过临时关闭可选卷积函数保持原 `nn.Conv1d + selective_scan`；候选才启用原 Mamba 的融合分支。记录实际 `mamba_inner_fn` 调用次数，比较完整更新耗时、每步梯度／更新量与重复误差。不预设会更快或数值相同。
- 原源码SHA、wheel及解包成员SHA绑定；运行时检查Torch2.7.1+cu126／Python3.11／CXX11 ABI TRUE。导入、ABI或算子执行失败则停止并返回紧凑错误，不自动安装或切换策略。官方wheel的原生导入和GPU融合前后向已由本次回传验证。
- 源Suite仍固定24981检查点，7个完成结果不重跑；原始训练继续暂停。临时数值／复制检查点沿用 finally 清理，离线依赖缓存保留供复用。自动回传 `qkan-mamba/mamba-convolution/<运行名>/mamba-convolution.json`；退出0表示测量完整和控制项一致，不表示数值准入或正式启用。
- 官方来源：[v1.5.2发布](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.2)、[CUDA接口包装](https://github.com/Dao-AILab/causal-conv1d/blob/v1.5.2/causal_conv1d/cpp_functions.py)。本地18项针对性标准库检查通过，包含离线解包／篡改拒绝、ABI前置拒绝、返回值转接、路径切换／异常还原、4子进程选择／失败／控制不一致与旧诊断兼容；无本地模型/GPU或依赖安装。

## 已完成实测：局部 BF16 无实用提速，停止重复该试验

2026-11-23 的 `mamba-projection-f7a88cb85e38` 已完成四个进程、18次一次性更新并自动回传。证据见 [局部投影原生审阅](../evidence/mamba-projection-native-review-20261123.json)。

- 完整更新中位耗时 FP32 **0.413508568秒**，输入投影BF16 **0.412347320秒**，吞吐仅增加 **0.2816%**；四组中三组更快，但未达到既有测速候选标准。两次两步的数值差异收窄：更新量相对L2约 **0.6128%～0.6459%**，原始梯度约 **1.4233%～1.4633%**；仍高于本次自身重复差异，不解释为收敛或数值准入。
- `causal_conv1d_available=false`，六块均报告 `original_unfused`。这确认当前可选卷积实现不可用，前次强制融合路径的要求与实际环境不匹配；原普通路径已完成本次计算。可选算子的缺失不等于已经证明它是耗时瓶颈。
- 初始状态、样本/RNG/时钟等控制一致；四个进程退出0且进程组清理完成；源step24981检查点未变，临时数组和副本已删除，正式提交0更新。原训练仍暂停，七个完成结果不变。
- 此局部BF16版本不作为正式加速入口，也不原样重跑。后续优先调查固定版本兼容的可选卷积／融合实现：先核对可用性与算子耗时，再设计保持FP32的短测速及数值对照。当前尚无融合候选的测速或准入结果；不安装依赖、不直接切换正式任务。

## 已完成：仅输入投影 BF16，保留原计算路径和内部 FP32

新候选 `mamba-in-proj-bf16-inner-fp32-v1` 只对每个 Mamba 块的 `in_proj.weight @ hidden_states` 使用 BF16，投影结果立即转回 FP32。投影偏置、卷积、x/dt 投影、选择性递推和输出投影保持 FP32；嵌入、LayerNorm、分类头、主参数及原 AdamW 不改。上游输入投影直接访问权重，包装 `Linear.forward` 会漏掉实际计算。

首次运行 `mamba-projection-2e3a7f239062` 的两个FP32子进程完成，候选在新增的类/参数/快路径兼容性检查处退出，尚无候选更新或测速；回传75074字节/SHA `7d7a1519b8fd870903f0ec12a46d2df96089e69a6afdb1664baa5fd29b9505da`。源文件未变，临时数组已清理。修复删除对快路径的额外要求：从固定源码提取原 `Mamba.forward`，只替换唯一输入投影表达式为 BF16→FP32 辅助调用，其余代码原样复用。因此沿用原生 `use_fast_path` 和可选 `causal_conv1d_fn` 分支，支持融合路径和原有普通卷积/scan路径，不安装依赖、不强制切换路径。报告新增每块实际路径、可选卷积可用性及表达式替换次数。错误摘要未区分旧兼容性检查的各子条件，不能仅从摘要认定具体缺失依赖。

仅在诊断进程中安装临时实例 forward，FP32 对照仍调用原 forward，退出时还原。绑定既有 Mamba commit `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` 的 `mamba_simple.py` SHA256 `a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3`；实际投影 BF16、内部输出 FP32 均有运行时检查。文件/唯一表达式边界/类或参数类型不匹配则停止，不修改安装包。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-5fdbeba08593 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_PROJ_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_PROJ_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_PROJ_REV:0:12}" &&
cd "$MAMBA_PROJ_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --projection-bf16 \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0
```

- 原检查点 step24981/SHA、micro8/logical32、dropout、样本时钟及 AdamW 分组保持固定。FP32 A/B 与局部 BF16 A/B 共四个全新进程、各两步；复用已有误差量化，返回每步更新量和原始梯度等相对L2/余弦/绝对误差，并比较自身重复误差、跨策略差异和精确控制项。
- 已完成的FP32报告保留；上次临时张量按既定 finally 策略删除，新测试必须重新生成同批FP32张量才可计算跨策略误差，不从哈希或标量伪造数组。旧全模型BF16/Graph/恢复测试继续复用。
- 最后一个进程追加两次预热和 ABBA×2 成对完整更新测速，共10次，合计预算18次一次性更新。每次从相同检查点/RNG/首批样本恢复，计时含 gather/H2D/前后向/裁剪/Adam/样本调度，排除恢复、哈希和数值复制。不重跑全模型BF16、Graph或恢复测试。
- 源 Suite 和七个完成结果保持原样，原正式训练继续暂停。每子进程900秒/阶段300秒；临时 FP32/局部BF16 数值文件与检查点副本沿用 finally 清理。仅回传 `qkan-mamba/mamba-projection/<运行名>/mamba-projection.json`。
- 退出0仅表示量化和测速完成、固定控制项一致；不等于数值准入。该候选的收益和误差尚待服务器实测，旧全模型 BF16 的45%提速不适用于它。若输入投影收益有限或误差仍大，应据实停下，不把 Graph 或更多低精度范围直接叠入正式训练。

## 已完成：量化三条路线的重复误差

FP32重复对照已完成，首步梯度/参数/Adam哈希不同，第二步损失仅差3.725290298461914e-9；BF16此前的重复误差幅度仍未知。使用现有入口新增的 `--measure-noise`：FP32、BF16 eager、BF16 Graph各两个独立进程，每个2步，固定共12步。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-06551e5475d4 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_NOISE_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_NOISE_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_NOISE_REV:0:12}" &&
cd "$MAMBA_NOISE_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --measure-noise \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "$QKAN_CACHE" --gpu 0
```

- 每步量化裁剪前累积梯度、裁剪后梯度、模型参数、Adam状态和本步参数更新量；报告整体范数、差异范数、相对L2、余弦、最大绝对差及按绝对差异范数排序的前8个参数。参考范数为零时相对L2/余弦留空，保留绝对差，不制造无限或任意巨大比值。
- 每条路线比较A与B的重复误差；两个BF16重复各自另与FP32 A比较，分清路线自身波动与跨精度偏移。只有两个重复、两个批次，属于描述性测量，不是统计噪声上界、收敛证据或可直接移植到其他任务的阈值。旧测速/8步损失证据继续复用；旧哈希报告缺原始数值，无法替代此次量化。
- 仍要求相同检查点SHA、初始科学状态、每步样本和位置、RNG、采样器、样本学习率时钟、Adam步计数及非浮点控制字段一致。原micro8/logical32/dropout/AdamW分组等不变；不进行新恢复测试，不覆盖原正式Suite或七个已完成结果。
- A运行的原始数值仅存本次Suite中3个已声明临时文件（每个128MiB上限），保存后重开并核对完整值/哈希；B读取带请求hash/创建模式绑定的文件再比较。全部临时数值及复制的源检查点在finally清理，不上传原始数组。每子进程900秒/阶段300秒，沿用独占空闲GPU、源锁与进程清理。
- 自动回传 `qkan-mamba/mamba-noise/<运行名>/mamba-noise.json`。退出0仅表示6个进程完成且上述控制项一致；`numeric_acceptance`始终未评价、`formal_release_allowed`始终false。数值很大也如实报告，不能将采集成功解释为数值通过；不完整或控制字段不一致返回76。`--measure-noise`与`--fp32-repeat`互斥，历史入口仍保留。

## 已完成：仅 FP32 重复性对照

已有回传表明两条 BF16 路线在中断前第2步就出现独立运行差异，中点加载本身完全一致。使用同一入口的 `--fp32-repeat`，只跑两个全新 FP32 进程、每个2步，总共4次一次性更新，保持原micro/logical/sample-clock/dropout。每步返回完整状态组件hash和损失，能发现“损失相同但梯度/参数已不同”。不启动BF16或Graph，也不重跑旧测速/40步验证。

在服务器执行整段（使用绝对路径，任意当前工作目录均可）：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-ff58cd8feab6 &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_REPEAT_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_REPEAT_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_REPEAT_REV:0:12}" &&
cd "$MAMBA_REPEAT_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --fp32-repeat \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "$QKAN_CACHE" --gpu 0
```

检查点SHA必须等于已审计BF16报告的输入（step24981），源若已推进则停止，不回滚源。两个进程初始完整状态及每步样本/损失/组件都精确一致才返回0；观察到差异返回76，同时显示首次差异步并自动回传 `qkan-mamba/mamba-repeat/<运行名>/mamba-repeat.json`。控制完成与重复性通过有区别，精确一致也不解锁BF16。新报告含旧审计hash以复用BF16证据；每子进程900秒/单阶段300秒，独占空闲GPU、源锁、退出清理沿用原实现。临时复制的检查点自动清理，原正式Suite和七个完成结果保留。

已审计测速见 [实测记录](../evidence/mamba-speed-native-review-20261123.json)：BF16 eager 吞吐 +45.22%，BF16 Graph +48.91%，两者单更新量相对差异约22.86%。本入口只补充数值和恢复证据，不重复四路线计时，不将Transformer阈值或K偏置策略用于Mamba。

## 执行指令

原训练目前保持暂停，无需再次 Ctrl-C。若已经恢复，则在原一体看板请求安全暂停并等入口退出。执行整段：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a33420b5408d &&
bash scripts/prepare_adaptive_checkout.sh &&
MAMBA_CHECK_REV=$(git rev-parse FETCH_HEAD) &&
MAMBA_CHECK_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MAMBA_CHECK_REV:0:12}" &&
cd "$MAMBA_CHECK_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mamba-validate -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-14d681111173/runs/bf16-graph-formal-20261119T014456Z-1cb7160c \
  --cache "$QKAN_CACHE" --gpu 0
```

新 worktree 复用现有环境与私密配置。原正式代码、检查点、七项已完成结果和其他模型均不变。只复制一个最新暂停的 Mamba 检查点；旧测速报告复用已提交的审计记录，不要求重传或重跑旧诊断。

## 固定预算和证据

- 固定接下来的8个完整训练更新，保留原物理micro和逻辑batch、不升档，保留dropout/AdamW分组/动量/样本学习率时钟/采样顺序。若当前epoch剩余行不足8步，明确停止，不换样本或偷偷跨epoch。
- 七个独立子进程：FP32参考8步；BF16 eager连续8步、4步中断、新进程恢复后4步；BF16 Graph同样8步与4＋4步。合计40次一次性更新，正式提交0步。每个子进程900秒上限、单阶段300秒上限，出错停止，不自动重复。
- 对每种BF16路线，记录相同8个训练批次的损失和行身份hash；第一步记录全部参数的原始累积梯度（裁剪前）及AdamW更新量，返回逐参数范数、差异、相对L2、余弦和最大绝对差。缺失梯度单独标记。原始数组不回传。
- 恢复检查比较连续8步与4＋4步的完整模型、optimizer、scheduler、RNG、梯度、permutation和epoch/cursor/step的hash、每步行身份及损失，并核对新进程起点等于中点。严格相等失败会显示失败，不降低规则。
- Graph捕获先于每个新进程的状态恢复；新进程恢复后继续原参数和随机状态。原训练的完整恢复还需实际worker接入验证，本次不是正式准入。
- 所有模型计算只访问train；val只建立原checkpoint契约所需的文件身份，未执行val/test前向。没有新的sample-trace.v2，不用此报告替代正式trace证据。
- 快照输入使用独立索引副本，避免推进8批次时改写原permutation视图。

回传 `qkan-mamba/mamba-validation/<运行名>/mamba-validation.json`，包括逐参数数值摘要、7个子进程退出/清理记录、连续恢复比对和源文件未改动结果。临时参考数组、中点状态和本次复制的checkpoint在finally清理，原源Suite权重始终保留。精简receipt和报告留存；原始异常文本与worker日志仅本地保存。

完成提示只表示固定前缀和恢复检查结束，不表示BF16已通过收敛或正式准入。对逐参数差异和8步损失进行数值审阅后，才决定实际worker接入；Image/Pathfinder输入形状另作短验证，MEGA随后，S5/S4D分别判断。

仅重传，不重跑：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory runs/实际mamba-validation目录 \
  --experiment mamba-validation --file mamba-validation.json
```
