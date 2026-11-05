# 本地科研门户

固定端口 4191，绑定 127.0.0.1。不要公开暴露本地门户，也不要把仓库根目录挂载为静态文件根目录。

```bash
python scripts/refresh_portal.py
python scripts/serve_portal.py --project-root . --portal-root research-web --port 4191
```

页面源数据来自版本化计划、审计、历史结果清单和说明文件。`registry.json` 是明确入口白名单，ready 表示页面可展示，不代表数据资格、实验完成或服务器在线。

采用 research-editorial-v1；表格在内部滚动、固定表头与首列、同步顶端水平滚动条，并支持排序、数据集筛选、可聚焦的列解释。
