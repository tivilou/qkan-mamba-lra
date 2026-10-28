# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned, tested baseline runners to the collaborator, with reproducibility and return-artifact contracts.

## Current State

- One-command candidate implemented: `scripts/run_baselines.sh`, `experiments/baseline_delivery/`, handoff `docs/baselines/one-click.md`. Preparation, independent GPU jobs, full-state resume and immutable chunked return are implemented; not yet certified.
- Engineering candidates: Transformer, explicitly pedagogical minimal upstream S4D, pinned Mamba1. Formal comparisons remain gated by foundation D01, cohort freeze and per-model fidelity.
- Primary-source inventory pins S4/S5/MEGA/Mamba and one explicitly unofficial LRU implementation. Candidate shortlist is provisional, not a current top-x claim.
- Legacy runner defects and per-model fidelity/recovery/trace delivery gates are documented in docs/baselines/delivery-contract.md.

## In Progress

- Supporting R01: two pages of recent primary-paper search and six detailed papers screened; B2S6/strong Transformer recipes and newer gates must be considered. See docs/baselines/recent-screen.md.
- No model implementation or formal run has been certified.
- Engineering diagnostics written: source-adapter mechanisms, inputs, fresh-process exact recovery and real sample-trace.v2 structural/semantic/replay. Tests will run through the shared queue.
- Upstream Apache-2.0 source ZIP built locally with hashes and license; exchange publication pending. Data is acquired directly from author, not redistributed through our exchange.

## Next Actions

- Register and execute the exact-source bounded CPU diagnostics on the original authorized node.

## Blockers

- Dataset qualification is insufficient solely on the remaining source/use evidence category; cohort and model fidelity remain uncertified. `release.json` is not_released; no formal-ready launch package.
- SSH has been intermittent; a persistent session on the authorized node is now established. Preserve other projects; do not switch to P4 without project-specific authorization.

## Validation

- Official source commits and README/paper metadata retrieved through the authorized server; compact source receipt committed with this stage.
- New runner has only static inspection so far; no new CPU/GPU PASS or asset-publication claim.
- At 52257f2, 31 qualification/data/pixel-source/exchange tests passed through the shared scheduler; all Image/Pathfinder pixels and labels exactly match pinned source. These tests do not certify baseline fidelity; no baseline performance is claimed.

## Relevant Decisions

- None.

## Plan Control

- Plan: plans/one-click/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: V01
- Focus Status: active
- Completed Steps: I01
- Next Action: 在原授权节点注册并执行有界 CPU 测试
- Blocker: 原授权节点 SSH 间歇 connection_closed，正在恢复稳定连接
- Last Checkpoint: 2026-10-28
