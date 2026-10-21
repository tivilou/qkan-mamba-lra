# 合作者实验交付

仓库：`tivilou/qkan-mamba-lra`。当前正式模型实验包尚未发布；旧训练命令仅用于历史开发参考，数据资格与基线协议完成后再发放正式运行命令。

## 一次实验的交接

1. 我们交付固定 Git 提交、配置、数据哈希、环境、启动命令、预算、恢复说明和返回文件清单。
2. 合作者在自己机器检出该提交，先通过硬件预检。按指定种子和完整数据运行，不自行改测试划分或挑选最佳测试轮次。
3. 保留独立运行目录。计算完成后，按实验白名单建立文件清单，使用仓库内工具上传大型内容。
4. 检查每个上传回执的文件大小与 SHA-256；把无密钥、无样本敏感内容的小摘要和回执清单通过 PR 返回。
5. 我们验收代码/配置/数据/运行身份、文件哈希和指标，再分析 Case Study、更新结果页与研究记录。缺文件时保留待补状态。

## 文件交换客户端

使用当前项目兼容的 Python 3.10+，无需第三方依赖。路径相对仓库根目录，与执行命令所在目录无关。

通过安全渠道获取 `PROJECT_EXCHANGE_URL`、`PROJECT_EXCHANGE_TOKEN`；私有 CA 如需使用，设置 `PROJECT_EXCHANGE_CA_FILE`。不要将这些配置放入 Git、PR、日志、浏览器 URL 或门户快照。

```bash
# 先生成本地清单；只列出本次实验明确允许返回的文件。
python scripts/exchange_artifacts.py prepare --experiment baseline-smoke --run-id run-001 --file runs/run-001/summary.json=summary.json --manifest runs/run-001/delivery.json

# 上传前校验；失败后用同一命令补传，不重新训练。
python scripts/exchange_artifacts.py upload --manifest runs/run-001/delivery.json
```

`delivery.json` 保留每个文件的相对路径、大小、哈希和上传状态，不包含服务地址或令牌。复制它到小型报告目录并检查隐私后再通过 PR 提交；`runs/` 默认不进入 Git。

文件服务按 `qkan-mamba/<experiment>/<run-id>/` 隔离。每个文件的上限由健康接口给出。大于上限的文件应先采用声明过的分片方案；当前工具明确拒绝超限，不静默拆分。已有路径不同内容必须冲突，同内容可安全重试。

## 已验收的基础设施

2026-10-21 已在明确选定的替代服务器恢复独立的 `qkan-mamba-exchange` 服务，使用 HTTPS、私有 CA、专用低权限账号与 `qkan-mamba` 命名空间。当前单文件上限 1 GiB。客户端流式传输，不将整个权重读入内存。

真实验收记录为 `docs/evidence/exchange-acceptance-20261021.json`；仅使用无模型结果的测试文件。上传、下载字节/哈希、相同内容重试、无授权下载拒绝、跨命名空间拒绝和不同内容冲突均通过。7 项本地集成测试另覆盖失败重试、错误回执、文件变化、路径安全和超限。尚未发生合作者正式实验交付。

管理员通过已授权 SSH 读取服务器 `/etc/qkan-mamba-exchange/` 内的令牌与 CA，按安全渠道发放；不要提交其内容。服务由独立 supervisor 程序 `qkan-mamba-exchange` 管理，应用位于 `/srv/qkan-mamba-exchange/app/`，产物位于 `/srv/qkan-mamba-exchange/storage/`。维护只针对本程序，不重启其他项目服务。服务模板和哈希来源见 `tools/exchange/PROVENANCE.md`。

## Case Study 和研究反馈

样本选择在运行前固定，基线和干预采用相同样本；按协议返回原样本、输入表示、门控输出、关键模型表示、预测与评估。缺失状态明确标注，不从最终指标构造隐藏表示。

原始大产物保留在认证服务，门户只展示经审查的摘要和受控数值投影。无任务收益或运行失败同样返回证据，不只提交最优结果。
