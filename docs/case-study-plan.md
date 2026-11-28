# 样本轨迹采集规划（待正式协议冻结）

状态：planned，尚未通过正式实验预检。本文件不宣称已有样本轨迹。

历史实验未保存原样本、logits、门控或内部表示，Case Study 状态为 unavailable：`trace_incomplete` 与 `provenance_incomplete`。不要使用历史模型权重猜测原运行内部状态。

新实验在数据资格核验和基线队列冻结后，固定数据命名空间、样本选择 seed、独立样本 ID、所需 train/validation/test 分割、training/evaluation/replay lanes、初始/最佳/最终模型状态。

按实际架构声明完整的关键表示清单：预处理后的 token/像素序列与 mask、embedding、各所选层的门控输出与 SSM 输出、pooling 表示、分类 logits、预测。每项声明 shape/dtype/坐标语义、生产阶段和上下游引用；不适用的阶段标 not_applicable。

目标标签仅用于允许的训练 loss 或评估，不进入推理特征。正文优先展示合法可见的真实输入与预测；原始文本或图像须确认许可。对固定样本保留全部所需数值张量，存于认证文件服务，并提供按坐标/切片读取真实值的受控投影。

交付前必须有 runner 发出的真实 sample-trace.v2、结构校验、项目语义校验和同样本确定性重放证据。页面存在或 JSON schema 通过均不能替代完整性检查。
