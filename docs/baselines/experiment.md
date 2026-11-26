# 统一实验入口

日常只使用`scripts/experiment.py`。它负责路由、原美化看板、父进程监督和安全暂停；科学执行器仍由Suite绑定的原版本负责。不是把模型、准入、诊断和回传实现合成一个大脚本，也不改变已有实验轨迹。

## 当前训练顺序：Retrieval 统一放最后（2026-11-25）

所有模型先完成 ListOps、Text、Image、Pathfinder，再处理未完成的 Retrieval。保持原模型顺序及其余数据集的相对顺序；已有结果直接沿用，暂停的 Retrieval 使用自己的最新完整断点。只改变派发顺序，不改变 BF16/Graph/融合策略、batch、学习率或训练预算。

如果原终端仍在训练，先按一次 Ctrl-C，等待保存并显示“实验入口已退出”，再执行以下整段；原本已暂停则直接执行。锁仍被占用时入口拒绝重复启动。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
git fetch origin research/collaboration-foundation &&
mkdir -p .bootstrap &&
git show FETCH_HEAD:scripts/experiment.py > .bootstrap/experiment.py &&
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/formal_progress.py &&
.venv-baselines/bin/python -B .bootstrap/experiment.py resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last
```

仍是一个终端内训练并显示美化看板，不创建新 Suite。原执行 checkout 的受跟踪文件保持不变。控制层在持有锁时记录 `task-order.json`，绑定原清单哈希；只给派发函数提供排序视图，原校验函数继续读取原清单，未替换共享模块全局变量。正式续跑 Suite 默认采用该顺序；历史普通 admitted Suite 保持原行为。`watch` 只读取已记录顺序。

恢复原顺序：安全暂停后，将同一续跑指令末尾改为 `--task-order manifest`。该选项仅恢复清单派发顺序，不回退训练状态。改顺序不重复已通过的恢复诊断，也不重跑已完成任务。

## 当前速度优化版：一条命令续跑并显示

BF16＋Graph 实际 worker 接入现使用显式 `start --profile bf16-graph`，见[首次切换整段指令](bf16-graph-formal.md)。它自动完成实际 worker 恢复检查，再在同一终端续跑和显示；新 Suite 后续直接 `resume`。以下命令保留为原 BF16 eager Suite 的续跑入口。

适用于已存在的`key-bias-formal-20261117T141411Z-047505d9`，原执行checkout为`a3569140d43b`。若仍在只读观察器中，Ctrl-C先关闭观察器，再粘贴整段。该操作不需要先更新训练checkout。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a3569140d43b &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/experiment.py > .bootstrap/experiment.py &&
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/formal_progress.py &&
.venv-baselines/bin/python -B .bootstrap/experiment.py resume \
  --suite runs/key-bias-formal-20261117T141411Z-047505d9
```

这是**一个终端、一个入口**：后台的原正式父进程继续使用原完整检查点，前台使用项目原Dashboard／Terminal。旧版本每30秒的JSON进入`progress-parent.log`，终端只显示看板。Ctrl-C／SIGTERM／SIGHUP仅向本入口拥有的父进程请求安全暂停，等待原worker保存和清理后退出；再次使用同一命令恢复。监督器锁及原Suite锁阻止重复训练。忙卡、版本不符、检查点缺失或回传失败仍由原入口检查，绝不为显示绕过。

当前候选保持BF16注意力、FP32主参数／Adam、解析K偏置梯度、eager16／逻辑128和无Graph。已有两个结果保留，其余任务沿用原分派。续跑不重新迁移、不重复数值／恢复／稳定性／性能试验。

只把两份控制／显示脚本放入被忽略的`.bootstrap`；不执行`git pull`，不覆盖原绑定文件、不迁移已有Suite。新控制器与显示脚本必须来自同一次已发布交付并放在同一目录。用户自行运行的正式训练与本机标准库控制验证分开记录。

## 五种日常操作

MEGA FP32 FFT 补齐短测：统一入口为 `diagnose mega-fft`，见[实验机整段指令](mega-fft.md)。正常完成和回传后自动恢复原策略训练与美化看板。

BF16 策略＋Graph 的显式隔离叠加测试：见[完整指令与门槛](bf16-graph.md)，统一入口为 `diagnose bf16-graph`。它不改变已有正式 resume 的 eager16 策略。

以下在使用的固定执行目录中调用。新checkout可直接用`scripts/experiment.py`，旧checkout采用上述同目录的`.bootstrap`副本。

| 操作 | 命令形式 | 行为 |
|---|---|---|
| 启动 | `experiment.py start --cache CACHE --profile PROFILE` | 新Suite准备通过后同一入口继续训练与看板；不接受已有Suite作为新输出 |
| 续跑 | `experiment.py resume --suite SUITE` | 自动识别正式候选／历史已准入Suite，调用该目录原执行器；默认一体化看板 |
| 查看 | `experiment.py watch --suite SUITE` | 明确只读，不启动训练；`--once`只打印一次 |
| 回传 | `experiment.py upload --suite SUITE` | 只重传已有产物；不加载UI、不启动训练 |
| 诊断 | `experiment.py diagnose KIND -- 原参数` | 显式请求才运行；绝不在resume时自动重复 |

`--root`是全局参数，放在子命令前；默认当前目录。Suite要求直接位于该原执行目录的`runs/`下。训练续跑前验证Git HEAD与Suite执行版本相同且受跟踪文件干净，再由原执行器验证全部科学绑定。运行环境使用该原checkout已有`.venv-baselines/bin/python`，不在统一入口中创建环境或安装依赖。

`start`和`resume`支持`--display auto|inplace|append`、`--interval 1..60`。默认TTY原地刷新／非TTY可读快照，间隔5秒。模型日志留在每个job的`worker.log`；父进程日志路径每次退出都会打印，并在看板末行标明。新启动在Suite创建前将日志存入`.bootstrap/experiment-*.log`，创建后通过Suite的`display-heartbeat.json`记录实际日志路径。

## 启动策略代替历史包装链

| `--profile` | 原意图 | 新入口做法 |
|---|---|---|
| `formal` | 普通正式准备 | 原readiness执行一次准入，再进入原admitted执行器 |
| `adaptive` | 显存适配 | 显式adaptive参数，不调用多层Shell包装 |
| `dynamic` | 动态逻辑batch | 显式adaptive＋dynamic参数 |
| `performance` | FP32吞吐优化 | 显式adaptive＋dynamic＋performance参数 |
| `accelerated` | 编译／Graph候选 | 原有参数与恢复／性能门保持不变 |
| `parallel` | 历史Retrieval Graph路线 | 原有parallel参数与原准入保持不变 |
| `key-bias` | 当前解析K梯度路线 | 必须提供`--migrate-from`和`--worker-from`；先原零更新准备，再自动resume |
| `bf16-graph` | BF16＋按文档 Graph | `--migrate-from`原BF16正式Suite、`--graph-from`已审计报告；复制最新状态、实际worker检查与回传通过后自动resume |

`--prepare-only`只做该次准备，不进入长期训练。普通profile可指定新的`--output`；key-bias由其原模块分配Suite，监督器只解析该次日志中固定的`Formal continuation Suite:`行，不打印原始日志。源迁移仍须显式给出；改变策略不属于resume，不自动将原FP32与候选权重混合。

诊断KIND为`attention-update`、`parallel-attention`、`partitioned-attention`、`operator`、`precision`和`padding`，对应原六个诊断Shell的模块，保留原参数与预算。其他工程／数据工具继续作为内部工具，不伪装成正式训练。

## 本次精简边界

整理前`scripts/`共有44个受跟踪脚本（17Shell＋27Python），其中9个Shell涉及训练／准入／迁移、6个诊断、1个准备目录、1个旧版批跑。本次新增一个统一Python入口后为45个（17＋28）；**物理文件未删除，推荐给用户的操作入口从多个收敛为一个**。详见[入口目录](experiment-entrypoints.json)。

九个训练／准入／迁移Shell保留为历史兼容入口；本次不修改其字节，尤其不改当前候选绑定的启动Shell。`run_admitted.py`与`formal_progress.py`降为统一入口内部复用组件。历史quick train和run all留作旧开发路径，不推荐用于当前正式Suite。后续只有在调用引用、测试与版本绑定都核对后，再考虑物理移除；本次没有删除科学实现、产物或任何运行目录。
