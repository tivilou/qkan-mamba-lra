# BF16 解析策略＋CUDA Graph：隔离叠加试验

## 当前结论

**2026-11-18 原生诊断已通过并完成回传审计：在 BF16 eager16 基础上，BF16＋Graph16 中位吞吐再提高 5.08%，完整更新耗时减少 4.83%。** 四组配对全部为正；数值和独立进程恢复通过。正式入口仍使用 BF16 eager16，组合的实际正式 worker 接入尚待实现和验证。证据见[原生结果审计](../evidence/bf16-graph-native-review-20261118.json)。

本次测量绑定 `ad6654f`，Suite `bf16-graph-20261118T104322Z-8870aa95`，读取原正式目录最新 Step20325 检查点；不是历史 Step14268 来源。BF16 eager 中位 6.2870945092 秒／128样本，Graph 中位 5.9833264109 秒／128样本，分别约20.36和21.39 samples/s。四组配对吞吐增益为14.60%、2.03%、3.16%、2.48%；重复数较少，不据此估计置信区间或整队列 ETA，不与此前不同来源的绝对耗时混比。

同一批次八次计时的完整状态哈希恰好全部相同；连续2步与新进程1＋1步的九项诊断状态也严格相同。五个子进程均退出0且进程组清理完成。原正式 checkpoint、元数据及两份已完成结果保持不变；15,999,568字节的私有中点已删除，正式提交0更新／0样本。

下方命令保留为此次试验的可复现记录。**这个成功试验无需原样重跑**；下一项工作是独立接入实际正式 worker，并验证被改变的执行与恢复路径，再考虑正式开启组合。当前恢复正式训练仍用本页末尾的原 BF16 eager＋美化看板入口。

只比较同一最新 BF16 正式检查点下的两条路线：

| 项目 | 对照 | 候选 |
|---|---|---|
| 注意力／主参数／Adam | BF16／FP32／原 Adam 历史 | 相同 |
| K 偏置梯度 | 原解析零梯度策略，保留历史 moments | 相同，钩子在 Graph 外 |
| micro／logical batch | 16／128 | 16／128 |
| dropout／样本／学习率时钟 | 原 0.1／同一行／原样本时钟 | 相同 |
| 执行 | eager | 每文档独立的 forward＋backward Graph；FP32 head 在 Graph 外 |

不加入 `torch.compile`、融合内核、micro32、零 dropout、重新初始化优化器或新的科学超参数。两个编码器副本梯度按固定顺序合并到原参数，每次返回新存储；Graph 静态梯度不会成为跨 micro 累加的别名。共享 CUDA RNG 的两个前向串行，反向各用私有流与静态缓冲。捕获前后恢复完整 RNG。

单文档 Philox 保守预算为 4,292,608,000，小于 uint32 上限 4,294,967,295；两个文档一起捕获超限，因此不做整对文档的单 Graph。该预算只排除已知溢出，不替代真实捕获检查。实现复用历史已测的 `parallel_attention.stream_graphed_encoder`，不修改该模块或原策略的组合禁令。PyTorch 图与随机数规则参见[固定版本源码](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/graphs.py)。

## 执行：一个诊断入口，不动正式 checkout

**若正式任务仍在运行，先在原训练终端按一次 Ctrl-C，等待安全保存和退出。** 不强杀，不为试验回退到旧 step14268。诊断要求原任务状态 `paused`、三个原锁空闲及所选 GPU 无计算进程；任何不满足均停止，不自动抢卡或暂停他人。

在执行端粘贴整段：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a3569140d43b &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-bf16-graph-checkout.sh &&
bash .bootstrap/prepare-bf16-graph-checkout.sh &&
GRAPH_ROOT="/root/projects/qkan-mamba-lra-adaptive-$(git rev-parse --short=12 FETCH_HEAD)" &&
(
  cd "$GRAPH_ROOT" &&
  .venv-baselines/bin/python -B scripts/experiment.py diagnose bf16-graph -- \
    --source-suite /root/projects/qkan-mamba-lra-adaptive-a3569140d43b/runs/key-bias-formal-20261117T141411Z-047505d9 \
    --cache "$QKAN_CACHE" --gpu 0 --child-timeout 900 --stall-timeout 300
)
```

准备脚本末尾打印的历史 readiness 提示无需再执行；上面已直接路由到新诊断。复用现有本地虚拟环境／缓存／私密回传配置，不安装依赖。新 checkout 与正式目录分开，不 `git pull` 正式目录、不重新迁移正式任务。新诊断入口已并入统一 `experiment.py`，不增加新的实验 shell 包装脚本。

## 一次试验的门槛和结果

顺序启动五个受监督的独立子进程；失败即停，失败报告同样自动回传：

1. **parity**：原 dropout 下，同起点完整更新的 logits、所有梯度、参数增量、Adam、时钟、排列、进度和 RNG。浮点 `rtol=1e-5, atol=1e-6`；RNG 和离散控制严格相同。失败只返回字段路径／类型，不上传张量。
2. **continuous / interrupt / resume**：Graph 连续 2 步，对照独立进程 1＋1 步；完整更新输出和九项诊断状态哈希严格一致。私有中点保存后重新读取验证，仅此兄弟中点可恢复。
3. **timing**：同一基线完整状态恢复后，ABBA×2 对照 eager16／Graph16。包含 CPU gather、H2D、前向、CE、反向、解析钩子、clip、Adam、样本时钟和有限值检查；排除捕获、预热、状态恢复和哈希。候选每个计时区间核验 Graph 常驻、同一捕获代、16 次前向／16 次反向重放及 8 次副本刷新。两路线均保留已驻留 Graph 的显存；记录准备耗时，不能直接代表端到端长期收益。

完整通过共 16 个可丢弃优化更新、2,048 处理样本、256 个独立原训练行，正式提交 0 更新／0 样本。每个子进程最终恢复其完整初始诊断状态；原 checkpoint、latest 指针、Suite 元数据及两份已完成结果的文件哈希须不变。恢复通过即按哈希删除私有中点；若中途失败则记录本地中点保留状态，绝不当作正式续跑源。

`runs/bf16-graph-*/bf16-graph.json` 自动回传到 `qkan-mamba/bf16-graph/`；日志、数据和权重不公开上传。回传失败保留已有报告及重试清单，不重复计算。报告 `passed=true` 还要求 **中位吞吐增益 > 0 且四个配对均 > 0**。Gate rejected／OOM／捕获失败都保留紧凑诊断，不降级后宣称 Graph 生效。

此处使用的九项状态是复用的数值诊断适配器，不是正式 worker 的新 checkpoint 格式。即使本试验通过，正式 worker 的组合接入、真实 worker 恢复及长期效果仍需后续各自验证，不用诊断结果冒充正式准入。

## 试验后继续原正式任务（带美化看板）

不论 Graph 是否提速，原正式任务仍按原路线继续。确认诊断已退出、GPU 已空闲后：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-a3569140d43b &&
.venv-baselines/bin/python -B .bootstrap/experiment.py resume \
  --suite runs/key-bias-formal-20261117T141411Z-047505d9
```

这是原一体化训练＋看板入口，不使用诊断中点，也不修改原 `training_strategy.graph=false`。如果此前没有保存统一入口副本，先按 `experiment.md` 的同一次 fetch 下载两份控制／显示文件，再运行同一 resume。
