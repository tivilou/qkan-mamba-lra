# 基础协作交付验收

日期：2026-10-21。交付分支 `research/collaboration-foundation`，首个实现提交 `f0be2c44ec785b255ce88621755d1be84e12eacc`，草稿 PR：https://github.com/tivilou/qkan-mamba-lra/pull/1 。

- 同一套 7 项标准库集成测试在本地 Windows 和替代服务器 Linux 均通过。命令：`python -B -m unittest discover -s tests -v`。
- Linux 从提交中的源记录运行 `python -B scripts/refresh_portal.py` 后，`git diff --exit-code -- research-web` 通过。报告字节与模板哈希使用 LF，内容未改写。
- 独立 HTTPS 服务的 6 项真实检查与 57 项桌面/移动浏览器断言分别记录在 `docs/evidence/exchange-acceptance-20261021.json` 和 `docs/evidence/portal-acceptance-20261021.json`。
- 替代服务器已恢复仓库到 `/home/qkan-mamba-lra/`；独立文件服务由 supervisor 维持运行。原始数据与历史 JSON、权重仍未恢复。
- 本轮完成 F01/F02/F03 基础阶段，当前焦点 D01 为五任务数据恢复和资格核验。相关工作调研、基线忠实复现、正式训练及真实 Case Study 仍未完成。

测试数据是文件交付夹具，不能作科研结果。令牌、CA 私钥、服务器连接配置、数据和权重均未进入 GitHub PR。
