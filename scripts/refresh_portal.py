#!/usr/bin/env python3
"""Project-specific read-only projections. Scientific source records remain owners."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "research-web"
PLAN = ".ai-progress/workstreams/foundation/plans/research-cycle/v001.json"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(path, value):
    target = PORTAL / path
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(target)


def main():
    plan = load(PLAN)
    results = load("docs/evidence/historical-results.json")
    for row in results["baselines"]:
        source = ROOT / row["provenance"]["source_ref"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["provenance"]["source_sha256"]:
            raise ValueError("Historical report changed: audit before refreshing its projection")
    apps = [
        ("overview", "research-system-orchestrator", "研究总览", "当前研究问题、已知证据与优先下一步"),
        ("plan", "workflow-plan-manager", "计划与进度", "稳定步骤、完成条件、证据和单一下一步"),
        ("qualification", "benchmark-dataset-qualification", "数据与相关工作", "五任务恢复与核验状态、调研范围和基线复现门槛"),
        ("experiment-results", "experiment-results-workbench", "实验结果", "Idea、跨数据集 Baseline 与 Case Study；明确历史证据边界"),
        ("research-notes", "author-facing-research-log", "研究记录与解释", "历史现场核查、经验与下一方向；没有编造已完成Idea"),
        ("collaboration", "collaborator-artifact-exchange", "协作与交付", "代码测试—合作者实验—认证回传—验收分析的操作契约"),
    ]
    registry = {"schema_version": "research-web.portal-registry.v1", "project": {"id": "qkan-mamba", "name": "QKAN–Mamba", "privacy": "127.0.0.1 only", "subtitle": "长序列分类 · 研究证据与协作工作台"}, "apps": []}
    for app_id, skill, title, summary in apps:
        app = {"id": app_id, "skill_id": skill, "instance_id": "qkan-mamba-" + app_id, "title": title, "summary": summary,
               "entry": "/apps/" + app_id + "/", "mode": "static", "status": "ready", "updated_at": "2026-10-28",
               "design_system": "research-web-design-system", "theme": "research-editorial-v1", "components": ["research-web-table.v1"]}
        if app_id == "experiment-results":
            app["components"].insert(0, "experiment-results-workbench.v1")
        registry["apps"].append(app)
        write(f"apps/{app_id}/web-app-manifest.json", {"schema_version": "research-web.web-app-manifest.v1", **app})
        html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · QKAN–Mamba</title><link rel="icon" href="/favicon.svg"><link rel="stylesheet" href="/assets/research-editorial.css"><link rel="stylesheet" href="/assets/site.css"></head><body><header class="top"><a href="/">QKAN–Mamba / Research</a><span>127.0.0.1 only</span></header><main data-app="{app_id}"><p class="eyebrow">RESEARCH WORKBENCH</p><h1>{title}</h1><p class="lead">{summary}</p><div id="content" aria-live="polite">正在读取证据…</div></main><footer>只读研究投影 · 来源和缺失证据在各页面明确标注</footer><div id="tooltip" role="tooltip" hidden></div><script src="/assets/workbench.js" defer></script></body></html>'''
        (PORTAL / f"apps/{app_id}/index.html").write_text(html, encoding="utf-8")
    write("registry.json", registry)
    write("apps/plan/data.json", {"plan": plan, "source": PLAN})
    write("apps/experiment-results/data.json", results)
    docs = {"overview": ["docs/audits/recovery-20261021.md", "docs/research-plan.md"],
            "qualification": ["docs/data-and-literature-readiness.md", "docs/dataset-qualification-v3.md", "docs/baselines/candidate-inventory.md", "docs/baselines/recent-screen.md", "docs/data-use-boundaries.md", "docs/case-study-plan.md", "docs/pathfinder-source-audit.md"],
            "research-notes": ["docs/research-log/README.md", "docs/research-log/20261021-date-only-foundation-audit.md"],
            "collaboration": ["docs/collaboration.md", "docs/baselines/delivery-contract.md", "docs/baselines/one-click.md"]}
    for app_id, paths in docs.items():
        write(f"apps/{app_id}/data.json", {"documents": [{"title": (ROOT / path).read_text(encoding="utf-8").splitlines()[0].lstrip("# "), "source": path, "text": (ROOT / path).read_text(encoding="utf-8")} for path in paths]})
    source = ROOT / "docs/text_task_pilot_results.md"
    shutil.copyfile(source, PORTAL / "apps/experiment-results/historical-report.md")
    print("Refreshed six registered workbenches from explicit source records")


if __name__ == "__main__":
    main()
