# MEGA 速度候选核对

- 截图：2026-11-26 10:45 CST，原89353fc/Suite b57105a1，一体化看板采用 Retrieval 后置；9已计算、mega-listops运行、14排队、Mamba Retrieval暂停。MEGA micro32/logical128，长期85.88samples/s、GPU99%、17.58/24GiB。截图不等于在线核验。
- 代码：models.py 显式 chunk_size=-1、未传 truncation；data.py ListOps 长度2048；requirements-mega.txt 固定 Torch1.13.1+cu117。不能直接套用新 Torch compile/SDPA 参数。
- 官方固定源码 aeaa4b44592cd1d60a9a34554e359eda2a62b03b 的 mega_sentence_encoder_layer.py 确认 bidirectional=True；exponential_moving_average.py 的 kernel_size=L、FFT长度为2*(L+K-1)，训练每次重算依赖参数的核，不能跨更新缓存核。
- 优先候选：实例插件将FP32 FFT补零长度8190改8192，保持核、输入补零位置和输出切片一致；卷积数学输出可保持，浮点结果仍可能不同，需比较输出/梯度/完整更新和原算法自重复差异。候选尚未实现、无实测速比；EMA耗时占比未知，不能预测整步收益。
- 后续条件候选：注意力/投影局部BF16而EMA保持FP32；现代融合注意力需要独立兼容环境和relative bias/mask/dropout语义核对。不能把改变chunk_size当作同模型纯执行优化。
- 本轮只读核对代码和官方源码，未启动模型/测试、未暂停当前训练、未修改训练代码/环境。仅记录调查结果。
- 来源（只读Invoke-WebRequest访问；web工具端点404后改直连，未下载文件）：
  - https://raw.githubusercontent.com/facebookresearch/mega/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/fairseq/modules/exponential_moving_average.py
  - https://raw.githubusercontent.com/facebookresearch/mega/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/fairseq/modules/mega_sentence_encoder_layer.py
  - https://raw.githubusercontent.com/facebookresearch/mega/aeaa4b44592cd1d60a9a34554e359eda2a62b03b/fairseq/modules/moving_average_gated_attention.py
