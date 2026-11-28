# 已保存参数的 Q/K/V 偏置分析

原归因报告显示四层 `self_attn.in_proj_bias` 占 Adam 更新误差平方和的98.3495%。本入口读取已完成实验的私有完整参数，按 PyTorch packed projection 的 Q/K/V 顺序各取128维，输出每层与四层合并的分段统计。K偏置的 softmax 抵消残差及小 Adam 分母是待验证假设。

本次只做 CPU 文件分析，不构建模型、读取训练 checkpoint 或数据缓存，不执行 forward/backward/optimizer step。原训练无需暂停，GPU无需空闲。使用现有离线 `.venv-baselines`，无依赖安装。只接受已审阅的 `ebe7e4732dc0` 归因结果：报告 SHA256 `ce784d88f6b9269c5f321f2a890b15c3a9bd2060b0d8d630bc6216a646f7ef44`，参数文件45431227字节、SHA256 `b0d0c8797a79865314ab37761ff52b3da0ef2f160c56994b8ad5aefd9bc6154f`。前后复核两个输入的哈希；原目录只读。

## 执行

在师弟服务器原 tmux 中执行：

```bash
cd /root/projects/qkan-mamba-lra-adaptive-b07d98e17ad0 &&
bash scripts/prepare_adaptive_checkout.sh &&
QKAN_BIAS_REV=$(git rev-parse FETCH_HEAD) &&
cd "/root/projects/qkan-mamba-lra-adaptive-${QKAN_BIAS_REV:0:12}" &&
.venv-baselines/bin/python -B -m experiments.baseline_delivery.attention_bias_postprocess \
  --attribution-suite /root/projects/qkan-mamba-lra-adaptive-ebe7e4732dc0/runs/attention-attribution-20261116T024408Z-e40dd283 \
  --child-timeout 900 --stall-timeout 300
```

准备 checkout 的通用提示仍会显示 GPU readiness 命令；本次执行上面的 CPU 模块即可。入口自动加载已保存的私密交换配置，在新 `runs/attention-bias-*` 目录保存结果。CPU子进程屏蔽CUDA可见设备、限制单线程，复用900秒总时限、300秒无阶段推进时限及自身进程组清理，不接触其他训练进程。

## 统计与解释

每段汇总裁剪前后梯度、实际权重增量、Adam一二阶矩前后值的绝对范数、相对差异、符号差异、零值数和绝对值分位数。所有分段平方和必须重建上一报告对应的 packed bias 统计，再汇总其占全局更新误差的比例。

以保存的真实 `step_after/lr/betas/eps` 和更新后矩在float64中计算 `denom=sqrt(v_after/(1-beta2**step_after))+eps` 及 `adaptive=-lr*(m_after/(1-beta1**step_after))/denom`，给出分母比值、矩递推方程残差、自适应项路线差与实际增量路线差的残差。参数文件没有原权重，故该项不构成含权重衰减及原FP32舍入的完整step精确重建。使用保存矩的方程残差也包含原实现算术舍入。

分母分箱固定为参考FP32分母的 `[eps,2eps]`、`(2eps,10eps]`、`(10eps,100eps]`、`>100eps`，只描述误差所在位置。空箱分位数为null。分段从每层到全层覆盖全部1536个偏置元素，不删除任何参数来重算准入指标。

只自动回传紧凑 `attention-bias.json` 至 `qkan-mamba/attention-bias/<目录名>/` 并验证SHA256回执。原 `paired-parameters.pt` 保持私有；不回传逐元素数组或日志。失败回传类型和代码位置，私有日志留在本次目录。上传失败保留已有结果和pending manifest，可单独重试，无需重复计算：

```bash
.venv-baselines/bin/python -B -m experiments.baseline_delivery.return_artifacts \
  --directory /absolute/path/to/new/runs/attention-bias-RUN_ID \
  --experiment attention-bias --file attention-bias.json
```

`postprocess_complete=true` 只说明文件分析完整；`candidate_admitted=false`。原20%全局Adam门限、BF16默认关闭和正式训练准入保持原样。本报告不建立因果、收敛、性能、Graph兼容或sample-trace.v2结论。Owner控制验证使用QT2标准库与替身；真实参数统计待师弟回传审阅。
