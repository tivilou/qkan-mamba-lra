# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned, tested baseline runners to the collaborator, with reproducibility and return-artifact contracts.

## Current State

- Workstream created on 2026-10-28; formal comparisons remain gated by foundation D01.
- Primary-source inventory pins S4/S5/MEGA/Mamba and one explicitly unofficial LRU implementation. Candidate shortlist is provisional, not a current top-x claim.
- Legacy runner defects and per-model fidelity/recovery/trace delivery gates are documented in docs/baselines/delivery-contract.md.

## In Progress

- Supporting R01: two pages of recent primary-paper search and six detailed papers screened; B2S6/strong Transformer recipes and newer gates must be considered. See docs/baselines/recent-screen.md.
- No model implementation or formal run has been certified.

## Next Actions

- Audit priority recent candidates' source/protocols and freeze the cohort after D01 qualifies.

## Blockers

- Dataset qualification is insufficient. All candidate fidelity statuses are not_verifiable; no formal launch package exists yet.

## Validation

- Official source commits and README/paper metadata retrieved through the authorized server; compact source receipt committed with this stage.
- At 6fd9392, 27 qualification/data/exchange tests passed through the shared scheduler; v3 document isolation checks passed. These tests do not certify baseline fidelity; no baseline performance is claimed.

## Relevant Decisions

- None.
