# 合作者集中验收入口

本入口只执行有界工程验收，不启动全量训练，不更新正式发布标志。研究方未运行模型测试；本版仅完成语法解析与代码审阅。保留原 suite 和依赖环境，无需重装。

```bash
git pull --ff-only origin research/collaboration-foundation
.venv-baselines/bin/python -B -m experiments.baseline_delivery.acceptance \
  --suite runs/baseline-20261103T115633Z-fa88b308 \
  --cache "$QKAN_CACHE" --device cuda:0
```

## 执行内容

1. Transformer、minimal S4D、Mamba1 的独立任务适配路径对照。固定输入、权重和无 dropout 条件，对比 logits、loss、全部梯度和一次 AdamW 更新；保留 S4D 参数优化器元数据。使用实际上游模块，检查的是任务适配边界，不宣称重现上游论文训练配方或性能。
2. 从原 suite 中唯一匹配的 mamba1-listops-s42 和 transformer-retrieval-s42 canary 配置与有界数据构建新运行。原目录不覆盖；每项含完整运行、中断、新进程恢复、实际 trace 语义和回放。
3. Mamba1 CUDA float32 使用逐元素对称 rtol=1e-5、atol=1e-6 的工程恢复预算，与已有 FP32 trace 预算一致。RNG、样本顺序、步数、调度器、AMP、best_accuracy/best_epoch 和全部离散状态精确比较。其他模型维持精确恢复。这个政策不是从重复最大值估计的统计噪声上界，也不保证长程轨迹逐位一致。
4. 成功或失败均回传已生成报告、适配结果、恢复明细以及本入口显式声明的诊断日志。目录：qkan-mamba/acceptance/<acceptance-id>/。不上传数据、检查点或原始 sample trace。网络失败按终端命令补传，无需重算。

## 报告解释

adapter_equivalence_observed 是适配对照证据，不是正式 fidelity 证书。engineering_checks_complete 是该入口完成，不等于所有 task/model/GPU 都已预检，也不改变 formal_release_allowed=false。审阅通过后，才能将运行证据纳入逐模型证书；数据资格、cohort 冻结以及其余正式配置预检仍需完成。

若原 suite 中找不到唯一匹配配置，入口会停止并上传失败摘要，禁止偷偷改用另一份运行。
