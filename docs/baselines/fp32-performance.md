# FP32 数据流水线与有界性能采样

此版本先优化数据搬运，不启用 AMP、TF32、torch.compile 或新的注意力实现。保持现有逻辑32/64/128、物理档位、裁剪、Adam、FP32与样本学习率策略。自适应控制器依赖实际耗时，执行速度变化可能影响未来档位选择，不能承诺后续逐位轨迹相同。

Torch 与 S5 都在本批 GPU 更新之前提交下一批 CPU 预取。只预取当前 epoch 中最多128条，消费时验证实际采样ID为预取ID的前缀，升降档仍严格使用提交游标；多取但未消费的行不会推进训练进度。线程只执行 NumPy 数据准备，不接触模型、RNG或GPU。Torch另复用锁页缓冲区；上一更新的同步完成后才覆盖，锁页不可用时回退普通CPU内存。GPU整批预放置OOM时仍沿用物理切片回退。

每个正式任务首次运行的16次已提交更新生成不可变 `jobs/<任务>/performance-profile.json`。Torch记录CPU等待、主机整理、传输、回滚快照、前向、反向、裁剪/优化器/调度器的主机耗时及CUDA事件区间；复用已有更新末尾同步，不额外逐阶段同步。S5仅记录主机等待和包含既有阻塞的完整更新耗时，不伪造CUDA阶段数据。OOM更新保留总耗时与重试数，舍弃可能不完整的阶段事件。各区间重叠，不可相加当成总耗时；前16次包含热身，不构成受控提速对比。保存checkpoint和验证/测试未包含在阶段报告中。

训练调度器用单独进程自动回传紧凑数值报告至 `qkan-mamba/performance-profile/<Suite>--<任务>/performance-profile.json`，每个监督入口内每任务最多启动一次（上传工具内部重试3次）。上传不阻塞训练调度；暂停时会停止尚未完成的上传，保留可重试清单。后续恢复可再次发送同一不可变文件。没有上传数据、日志、权重或原始trace；最终整套回传仍包含此报告。

## 当前9a实验迁移指令

适用于已验收的源版本 `9a147cd70cedf1511de89d0381dfbdabac3274f2`，截图中的Suite为 `adaptive-migrated-20261106T111646Z-90465`。源checkout必须保持原版本。**在原监督入口按一次Ctrl+C，等待保存完成并返回shell提示符**，再执行整段：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-9a147cd70ced &&
test -d runs/adaptive-migrated-20261106T111646Z-90465 &&
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh &&
QKAN_NEW_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_NEW_REV:0:12}" &&
bash scripts/migrate_performance.sh \
  --migrate-from /root/projects/qkan-mamba-lra-adaptive-9a147cd70ced/runs/adaptive-migrated-20261106T111646Z-90465 \
  --cache /root/qkan-cache --gpus 0
```

入口生成新版本的独立checkout和新Suite，验证并复制未完成任务的最新checkpoint，保留模型、优化器计数/矩、RNG、采样排列/游标、历史/最佳权重、累计耗时、**原有sample_clock及controller（包括OOM上限和未完成窗口）**；不重新锚定已有学习率曲线。不修改源文件。五个源码探针、未完成任务的恢复/trace验收，以及已有断点的零更新原生恢复/序列化核验均成功后，入口自动继续剩余训练。没有公共网络下载与新依赖；复用既有环境/缓存/认证共享服务。

已完成任务现在支持原样保留：逐一核对原config、summary、正式运行标志、选中权重哈希、trace验收哈希、最新完整断点和保留的前一代断点，再复制原始字节至新Suite的同名任务目录。原始配置、summary、权重和版本身份均不改写。缺少或损坏任一必要产物会明确中止，绝不降级成从头训练。旧版模型无需再次导入或测试；原始私有trace和日志继续保存在源Suite。

新Suite仍显示25个任务；沿用的结果计为“已计算”，另有“沿用已完成结果（保留原版本，不重跑）”提示。例如现在已完成ListOps，会显示已计算1，只对其余24个任务执行新版验收与后续训练。迁移快照还保存原Suite/context/gate报告及现有gate-evidence/runtime-release字节，迁移报告明确列出preserved_completions，原结果不获得新版验收身份。最终自动回传包含原结果及未完成任务的新结果、迁移血缘和原验收快照；delivery.json用result_source_revisions逐任务标记版本，mixed_source_revisions明确混合版本，不能把整套结果宣称为新版统一重跑。

记下新入口打印的 `Suite:`。以后在同一新checkout中恢复：

```bash
.venv-baselines/bin/python -B scripts/formal_progress.py \
  --resume 新Suite绝对路径 --display inplace --interval 5
```

单独补传耗时报告（不重新训练）：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory 新Suite绝对路径/jobs/transformer-listops-s42 \
  --experiment performance-profile --run-id 新Suite目录名--transformer-listops-s42 \
  --file performance-profile.json
```

本机仅标准库/模拟检查，不导入框架、查GPU、装依赖或运行模型。原生验收和吞吐变化待师弟服务器确认。优先比较稳定档位的samples/s；拿到报告后再决定编译或混合精度的独立实验。
