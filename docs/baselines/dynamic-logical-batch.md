# 动态逻辑 batch 与旧断点迁移

这是独立的可选训练策略：逻辑 batch 在 **32 / 64 / 128** 间调整，物理 micro-batch 在不超过当前逻辑 batch 的 **1 / 2 / 4 / 8 / 16 / 32 / 64 / 128** 档位间调整。覆盖原生 PyTorch 与 S5/JAX；复用现有离线环境、数据和文件共享服务，不引入下载或新依赖。原有固定逻辑 batch 策略仍有独立入口。

## 控制规则

- 新控制器从逻辑32、物理4起步，按显存余量尝试下一档。物理切片的估计峰值上限为整卡85%，额外保留512MiB；缺少显存读数时继续训练，但不主动升档。
- 物理切片已等于逻辑 batch 时，可同时扩大两者；若物理切片受预计峰值限制但仍有至少15%的整卡空闲和512MiB余量，可以只扩大逻辑 batch、继续分片累积梯度。逻辑档位不受物理切片停留在32的限制。
- 比较单位是每个样本的完整更新时间，等价于 samples/s。每窗16次完整更新，切档后忽略2次预热，至少3个确认窗。任何确认的正向收益均可保留，没有5%的最低收益线；一次慢窗不回退。持续确认变慢则退回试档前的配置，冷却256次测量更新。最多8个比较窗仍不明确时保留候选并冷却，不宣称提速。
- 切档发生在完整更新边界。OOM先恢复模型、优化器、学习率时钟和RNG，缩小物理切片，重试**同一组样本**；整批成功后，下一批的逻辑 batch 再降一档并冷却。最多8次OOM重试；物理1仍OOM则明确失败。非OOM异常不会被伪装成显存问题。
- epoch末尾不足整批的样本仍完整训练，不参与吞吐试档计时；不会丢样本、补样本或跨epoch拼接。所有试档更新都计入真实训练预算。

控制器寻求更高吞吐，不强制填满显存，也不保证全局最优。逻辑上限128是本次有限策略的边界；不自动叠加并发、AMP、TF32或新编译器。JAX新形状可能重新编译，编译期间的热身读数不用于升档判断。

## 样本时钟与恢复

总预算为 `epochs × train_rows`，进度为 `epoch × train_rows + cursor`；每次成功更新后只增加这次实际样本数。改变逻辑 batch 后，**剩余样本预算保持不变，剩余optimizer更新次数可以减少**。实际Adam步数及一、二阶矩不改写、不按batch倍率缩放。梯度按当批实际样本数求平均、完整累积后统一裁剪。

学习率按已提交样本数调度，checkpoint保存 `sample_clock`。从旧断点迁移时，以当前各参数组学习率和样本位置为锚点继续，不重新预热，也不线性放大学习率；原有参数组比例得以保留。迁移点之后的曲线是明确的新策略，不能宣称与旧学习率曲线或固定batch优化轨迹等价。

PyTorch时钟随可回滚scheduler状态保存/恢复；S5保持不可变候选TrainState，成功后才推进主机样本时钟。S5学习率闭包在原有eager Optax更新路径中读取时钟，梯度计算独立JIT。checkpoint同时保存逻辑/物理档位、未完成比较窗、冷却状态、旧断点血缘和原生优化器状态。恢复时核对样本时钟与采样游标；旁路显示文件不能覆盖checkpoint状态。

显示界面沿用原地刷新，增加本步/下一步逻辑batch，保留micro-batch、GPU利用率和显存。动态策略显示已处理样本数、样本百分比、samples/s、实际Step及**当前任务剩余训练样本ETA**；不会继续拿旧300000步当分母，也不把它标成全部排队任务ETA。ETA不含后续验证、测试与回传。

## 将当前实验迁入新版本

当前已知源是 `adaptive-migrated-20261106T042749Z-67281`，位于执行器 `qkan-mamba-lra-adaptive-96e542dd527b`；支持该96e542d版本以及最初已验收的31639c0固定版本。源checkout必须仍匹配其验收版本且没有被改写的tracked文件。已完成的源任务不迁移、不重新标注；已启动但缺少完整checkpoint的任务拒绝从头降级启动。

在原监督入口按一次Ctrl+C，**等保存结束并回到shell提示符**，然后执行下面整段。不要先pull源执行器，不要同时运行两个实验。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-96e542dd527b &&
test -d runs/adaptive-migrated-20261106T042749Z-67281 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/migrate_dynamic_logical.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-96e542dd527b/runs/adaptive-migrated-20261106T042749Z-67281 \
  --cache /root/qkan-cache --gpus 0
```

一个入口自动完成：复制并验证源配置/checkpoint哈希 → 新版本五个源码探针和25个任务恢复/trace验收 → 原生worker零更新恢复与序列化核验 → 自动回传紧凑验收产物 → 启动原地刷新监督器，继续剩余训练。源文件、源checkpoint和源验收证书保持原样；新controller重置旧吞吐计时，保留完整的前序迁移血缘。

本次动态恢复canary在两个有界epoch中强制执行 `32 → 64 → 128` 两遍，第二次更新后中断，另一个进程恢复后执行128档。它检验原生变形状更新、优化器/样本时钟恢复和trace；吞吐决策的时间/显存输入另由标准库模拟检查，不能靠一次canary证明所有未来显存压力下的决策相同。强制档位仅允许在明确的合成验收模式中使用，正式任务不能开启。验收使用训练/验证的有界样本，不访问测试集。

保存入口打印的新 `Suite:` 绝对路径。之后恢复这个新实验，必须在同一新执行器中使用：

```bash
.venv-baselines/bin/python -B scripts/formal_progress.py \
  --resume 新入口打印的Suite绝对路径 --display inplace --interval 5
```

如果验收失败，保留新Suite；修复后用 `run_formal_readiness.sh --dynamic-logical-batch --resume 新Suite --cache /root/qkan-cache --gpus 0 --gates-only` 继续该次验收，再启动监督器。不要反复调用迁移包装脚本生成重复Suite。缺少migration-validation.json但已有目标checkpoint时会阻断，需检查完整性。

## 证据边界

用户截图显示当前旧自适应策略已达到逻辑32、物理32、约3.152 step/s、显存11.70/24GiB；这是用户提供的观察，不是受控提速对比。本次修改在本机只用标准库/模拟验证，未导入模型框架、查询GPU、安装依赖或启动训练。新原生恢复与性能由上述入口在师弟服务器验收，代码交付不等于已通过该验收。

动态batch改变优化轨迹，可能影响最终精度；不得直接当作固定batch基线结果。模型来源、数据身份和FP32保持明确，但吞吐收益、精度和恢复成功仍分别需要实测证据。源码和紧凑检查证据进入Git；数据、权重、原始日志和trace留在原存储/认证共享服务。
