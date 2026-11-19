# BF16＋Graph 正式续跑

当前变更把已验证的按文档 BF16 Graph 接到实际 worker；`worker.py`、完整更新体、Adam、样本调度、数值策略以及验证／测试／trace 路径均保持原实现。使用独立 checkout 和新 `bf16-graph-formal-*` Suite，从原 BF16 正式 Suite 的**实际最新完整检查点**复制续接，保留两个已完成结果及其全部字节，其余22项仍由原执行器运行。

本地标准库／fake 检查不代表已经在协作者 GPU 上通过实际 worker 检查。之前隔离诊断吞吐中位数增益为5.07691%；正式持续训练的实际速度由看板报告。

## 首次切换：整段执行，训练和看板一体

原正式任务应已安全暂停并回到 Shell 提示符。若仍在原训练入口中，按一次 Ctrl-C 并等待完整检查点保存、进程退出。保持原目录与原检查点；不在原 checkout 执行 `git pull`。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a3569140d43b &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_GRAPH_REV="$(git rev-parse FETCH_HEAD)" &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_GRAPH_REV:0:12}" &&
.venv-baselines/bin/python -B scripts/experiment.py start \
  --profile bf16-graph \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-a3569140d43b/runs/key-bias-formal-20261117T141411Z-047505d9 \
  --graph-from /root/projects/qkan-mamba-lra-adaptive-ad6654f72baa/runs/bf16-graph-20261118T104322Z-8870aa95 \
  --cache "$QKAN_CACHE" --gpu 0
```

代码准备工具只创建独立目录并链接已有环境，不安装依赖。它打印的历史 readiness 提示不需要另外执行；上面的最后一条才是本次统一入口。

入口自动校验已审计 Graph 报告及29个执行文件绑定、最新源完整状态和数据／配置合同；锁住原正式 Suite 与历史源 Suite，拒绝同时运行。新目录复制后先运行四个**实际原 worker** 的短程子进程：原 BF16 eager 连续2更新、Graph 连续2更新、Graph 中断1更新、新进程 Graph 恢复1更新及原评估 trace。合计6次临时更新，正式源0更新；临时检查点不会用于正式续跑。全状态浮点容差 `1e-5/1e-6`、RNG及控制精确；Graph 连续2与1＋1全科学状态精确一致。检查点、参数梯度、Adam、历史、最佳验证选择以及采样位置均纳入检查。

成功和失败均自动回传紧凑报告、哈希回执及可重试记录。四个子进程退出／进程组清理、临时代文件删除、实际 trace 验证和报告回传全部通过，才写入 `graph-ready.json`。同一命令随后自动进入正式训练，仍从未推进过的源副本开始；原美化看板持续显示，不另开观察器。不重复之前通过的隔离诊断、迁移或 best 重评分。

## 执行与显示

完整16条 micro 使用 Graph，逻辑 batch 128；不足16条的 epoch 短尾明确使用原 BF16 eager，验证、测试和 trace 也走原 BF16 eager。解析 K 梯度 hook 保留在 capture 外，FP32 参数／Adam 不变。Graph 执行谱系另存 `bf16_graph_execution`，原 `training_strategy.graph=false` 是冻结的数值策略历史字段，不将其重写为新数值策略。

看板通过**当前 PID** 的 `bf16-graph-execution.json` 显示捕获代数、实际前向／反向 replay、完整 micro 和短尾 eager 数；初始化和 preflight 显示准备阶段，旧进程记录不会冒充本次实际执行。Graph 捕获／形状／覆盖错误会保存紧凑诊断并结束，不静默换成未验收路径。此次没有 Inductor 编译融合。

## 再次续跑与撤回

首次启动会打印并在看板显示新 Suite 的绝对路径。此后在该新固定 checkout 执行（把 `SUITE` 换成打印的路径），无需再次 start：

```bash
.venv-baselines/bin/python -B scripts/experiment.py resume --suite SUITE
```

Ctrl-C 请求本入口拥有的子进程安全保存后退出。再次 resume 使用新 Suite 自己的最新 Graph 完整状态，不重复已通过的实际 worker 检查。已有报告重传用 `experiment.py upload --suite SUITE`，它不启动训练。

若撤回本次候选，先安全暂停新 Suite 并等它完全退出。原 `a3569140d43b` Suite 与检查点一直保留，可用原统一续跑命令恢复**原来暂停的游标**；这会建立另一条从旧游标出发的轨迹，不会自动带回新 Graph Suite 已推进的权重。禁止两条轨迹同时运行。代码回滚包只恢复本次交付文件，不改任何实验目录／检查点。
