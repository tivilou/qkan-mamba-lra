# MEGA/S5 接入状态（2026-11-04）

数据资格已在限定内部非商业研究、不再分发原文／数组的范围内通过；模型发布仍未通过。

实现名单固定为 Transformer、Mamba1、MEGA、S5，minimal S4D 为辅助机制对照。两个强参照覆盖 attention 与 SSM 家族，均有任务匹配的作者 LRA 源码／配方；这是有边界的选择，不声称全球前二。B₂S₆ 等近邻的工件缺口保留，结论不得外推到这些未实测方法。

本轮已实现 strong_sources.py：验证完整官方仓库的根路径、固定 commit、干净状态、关键源码哈希及实际导入位置，提供官方 MEGA 与 Flax S5 编码器构造入口；不下载、安装或切换框架，不静默回退。来源、机制与适配记录在 strong-source-registry.json 和 manifests/mega、manifests/s5。

这还不是可运行实验：当前 runner 的 --models 只支持原有三个候选。MEGA 需要 fairseq 兼容环境及任务／训练接口；S5 需要独立 JAX/Flax 环境、原生训练和全状态恢复／trace。作者 S5 GPU requirements 含未填版本占位符，不能原样交付自动安装。所有新模型运行 gate 均保持 false，证书状态 not_verifiable。

下一步具体交付物是两个框架各自的锁定环境、原生数据/head适配及worker、原子全状态checkpoint、sample-trace.v2、源机制比较和统一自动回传验收。完成后再交付一次实际设备命令；现在不要求师弟运行新命令。
