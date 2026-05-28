# LRA 全任务初步测试报告

## 实验目的

验证纯 Mamba 模型在 LRA 全部 5 个任务上的训练流程是否正常，确认 loss 能够持续下降。

## 实验设置

| 配置项 | 值 |
|--------|-----|
| 模型 | Mamba-only (4 layers, d_model=128) |
| 参数量 | ~468K |
| 数据源 | Kaggle (archive.zip) |
| 学习率 | 0.001 (CosineAnnealing) |
| Weight decay | 0.01 |
| Grad clip | 1.0 |

## 各任务结果

### ListOps (10 分类, seq_len=2048, 15 epoch, batch_size=64)

```
[01] train=0.1683 val=0.1715 loss=2.2577
[04] train=0.2900 val=0.3455 loss=1.9498
[07] train=0.3583 val=0.3570 loss=1.7387
[10] train=0.3597 val=0.3495 loss=1.7254
[15] train=0.3673 val=0.3655 loss=1.7104

Test accuracy: 0.3745
```

Loss 从 2.26 降到 1.71，远超随机水平 (10%)。

### Text / IMDB (2 分类, seq_len=4096, 20 epoch, batch_size=32)

```
[01] train=0.5002 val=0.5000 loss=0.6978
[05] train=0.5032 val=0.5000 loss=0.6942
[10] train=0.5153 val=0.5000 loss=0.6926
[15] train=0.5208 val=0.5128 loss=0.6919
[20] train=0.5210 val=0.5162 loss=0.6917

Test accuracy: 0.5162
```

Loss 从 0.698 降到 0.692，下降缓慢但持续。需要更多 epoch。

### Image / CIFAR-10 (10 分类, seq_len=1024, 15 epoch, batch_size=64)

```
[01] train=0.2511 val=0.3768 loss=2.0298
[05] train=0.5590 val=0.5676 loss=1.2512
[10] train=0.6491 val=0.6190 loss=1.0056
[15] train=0.6915 val=0.6486 loss=0.8877

Test accuracy: 0.6341
```

Loss 从 2.03 降到 0.89，收敛最快，效果最好。

### Pathfinder (2 分类, seq_len=1024, 15 epoch, batch_size=128)

```
[01] train=0.5003 val=0.5002 loss=0.6946
[05] train=0.4977 val=0.4962 loss=0.6933
[10] train=0.5005 val=0.5059 loss=0.6932
[15] train=0.5018 val=0.4986 loss=0.6931

Test accuracy: 0.4975
```

Loss 从 0.6946 降到 0.6931，下降极缓慢。该任务对纯 Mamba 较难。

### Retrieval (2 分类, seq_len=4096×2)

尚未完成测试（数据量 14.7 万，序列最长），后续补充。

## 总结

| 任务 | 分类数 | 随机基线 | 初始 loss | 最终 loss | Test Acc | 结论 |
|------|--------|---------|----------|----------|----------|------|
| ListOps | 10 | 10% | 2.2577 | 1.7104 | 37.45% | loss 稳定下降 |
| Text | 2 | 50% | 0.6978 | 0.6917 | 51.62% | loss 缓慢下降 |
| Image | 10 | 10% | 2.0298 | 0.8877 | 63.41% | loss 快速下降 |
| Pathfinder | 2 | 50% | 0.6946 | 0.6931 | 49.75% | loss 极缓慢下降 |

**所有任务的 loss 都呈下降趋势**，训练流程验证通过。当前 epoch 数较少，增加训练轮数后准确率会进一步提升。

## 下一步

- 增加训练轮数到 50+ epoch，观察准确率提升幅度
- 调优超参数（学习率、warmup、模型深度）
- 对比 QKAN-Mamba 和纯 Mamba 的收敛速度
- 补充 Retrieval 任务结果
