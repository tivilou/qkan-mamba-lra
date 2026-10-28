# 复现与合作者交付验收

我们负责源码、测试、协议和结果分析；师弟负责正式全量实验。当前只交付资格检查代码，**还没有可以启动全量 baseline 的已验收版本**。先通过数据资格，再冻结队列，最后逐模型通过以下验收。

1. **来源与机制**：每个模型固定 commit、依赖和文件哈希；提供 `baseline-manifest.json`、`mechanism-contract.md`、`adaptation-diff.md`，运行官方模块与 adapter 的前向、loss、梯度和一次更新比较。不能只检查 shape、import 或参数数目。
2. **输入协议**：ListOps/字符文本保留原始 token IDs，使用 embedding、padding mask 和声明的 EOS/pooling；图像固定 `float32(x)/255-0.5` 的逐像素 row-major 序列，不用训练/测试联合统计；Retrieval 两篇文档共享 encoder，pair head 与顺序对称性另行冻结。资格检查使用的 bag 特征不是正式序列模型输入。
3. **选择与恢复**：只用 validation 选择最优 checkpoint，test 从最优权重独立评估。latest 保存 model/optimizer/scheduler/scaler/RNG/sampler permutation/next-batch cursor/best 状态；通过真实子进程中断恢复、一致性及损坏拒绝检查。运行目录包含 task/model/seed/config 指纹，拒绝覆盖。
4. **可观察性**：冻结样本、输入、关键表示与输出契约；真实产生 `sample-trace.v2`，通过结构、项目语义、baseline/control replay 检查。样本级预测和隐私数据通过文件服务保存，GitHub 仅放审阅过的摘要及哈希。
5. **执行**：正式 GPU 预检须在师弟实际设备上核对 CUDA kernel、dtype、显存、吞吐与恢复保证。不可用 CPU 小样本代替 GPU 实测。先完整数据单种子；按预定规则再扩大种子数，不通过筛选有利 seed 改写结论。
6. **回传**：GitHub 放固定配置、环境清单、compact metrics、fidelity/preflight/recovery 报告、运行身份、文件清单和 SHA-256；文件服务放 checkpoint、详细日志、逐样本结果、trace。只接收预定字段与路径，核对源 commit、数据/配置指纹和全部回执后再分析。
7. **分析**：区分执行失败、假设无效和证据不足；所有对照使用同一数据版本，报告资源成本及不确定性。Case Study 读取预先固定的真实样本与中间量；更新结果页、研究解释与研究日志后决定继续、修改方向或接受研究目标完成。

旧 `experiments/lra/train.py` 不满足上述契约：token ID 连续化、配对直接拼接、checkpoint 跨任务覆盖、测试没有加载最佳验证权重、没有完整恢复。旧 `src/baselines/s4.py` 不能作为官方复现；保留历史代码供审计，禁止直接复用旧命令作为正式交付。

资格检查的固定命令（由统一调度注册 recipe 执行）：

```bash
python -B scripts/run_qualification_preflight.py --output-dir "$RESEARCH_OUTPUT_DIR"
python -B scripts/audit_lra_relations.py --data "$LRA_DATA" --scratch "$LRA_SCRATCH" --output "$RESEARCH_OUTPUT_DIR/relations.json"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B scripts/qualification_pilot.py --data "$LRA_DATA" --output "$RESEARCH_OUTPUT_DIR/pilot.json"
```

预检只使用合成数据；关系审计读取五任务版本中的 Retrieval 全部三划分但不训练；pilot 只取 train≤4096、validation≤1024，每任务固定 12 epoch 线性参考和 shuffled-label control，不打开 test。每个报告都保留 `formal_baseline_allowed=false`；弱参考没有信号只表示诊断不充分，不能否定长程序列任务可学习性。
