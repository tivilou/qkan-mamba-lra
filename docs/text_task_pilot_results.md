# LRA Text 任务初步测试报告

## 实验目的

验证纯 Mamba 模型在 LRA Text 任务上的训练流程是否正常，确认 loss 能够持续下降。

## 实验设置

| 配置项 | 值 |
|--------|-----|
| 模型 | Mamba-only (4 layers, d_model=128) |
| 参数量 | 467,714 |
| 任务 | Text (IMDB 字符级情感分类，二分类) |
| 序列长度 | 4096 (Kaggle) / 4000 (MEGA) |
| Batch size | 32 |
| 学习率 | 0.001 (CosineAnnealing) |
| 训练轮数 | 10 epoch |

## 数据源对比

分别使用 MEGA 和 Kaggle 两个数据源进行了测试，验证数据一致性。

### MEGA 数据 (seq_len=4000)

```
[01] train=0.4988 val=0.5000 loss=0.6979
[02] train=0.5015 val=0.5049 loss=0.6991
[03] train=0.5028 val=0.4951 loss=0.6956
[04] train=0.5100 val=0.4964 loss=0.6942
[05] train=0.5100 val=0.5090 loss=0.6931
[06] train=0.5162 val=0.5162 loss=0.6924
[07] train=0.5166 val=0.5132 loss=0.6921
[08] train=0.5196 val=0.5104 loss=0.6919
[09] train=0.5178 val=0.5167 loss=0.6919
[10] train=0.5219 val=0.5159 loss=0.6916

Test accuracy: 0.5159
```

### Kaggle 数据 (seq_len=4096)

```
[01] train=0.5012 val=0.5102 loss=0.6994
[02] train=0.4973 val=0.5000 loss=0.6977
[03] train=0.5035 val=0.5000 loss=0.6954
[04] train=0.5012 val=0.4982 loss=0.6951
[05] train=0.5038 val=0.5032 loss=0.6938
[06] train=0.5026 val=0.5002 loss=0.6932
[07] train=0.5099 val=0.5161 loss=0.6928
[08] train=0.5131 val=0.5082 loss=0.6924
[09] train=0.5187 val=0.5163 loss=0.6921
[10] train=0.5206 val=0.5157 loss=0.6918

Test accuracy: 0.5157
```

## 结论

1. **Loss 在持续下降**：从 ~0.699 降到 ~0.691，趋势稳定
2. **两个数据源结果一致**：最终 test acc 都在 0.515 左右，说明数据内容等价
3. **需要更多 epoch**：二分类随机基线 loss = ln(2) ≈ 0.6931，当前刚低于这个值，模型刚开始学到信号
4. **参考值**：论文中 S4 在 Text 任务上达到 ~86%，通常需要 50-100 epoch 训练

## 下一步

- 增加训练轮数到 50 epoch，观察准确率是否明显提升
- 调优超参数（学习率、warmup、模型深度）
- 对比 QKAN-Mamba 和纯 Mamba 的收敛速度
