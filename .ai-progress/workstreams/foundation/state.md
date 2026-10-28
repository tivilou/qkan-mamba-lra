# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- MEGA author-linked LRA archive is pinned and rebuilt into five adapted tasks on the replacement server; original source bytes and previous output remain preserved.
- Data pipeline revision `ac8e0d4` adds safe reconstruction, full-array diagnostics, conservative Retrieval pair curation and 13 passing tests. No formal model training launched.
- GitHub publication through `6947b0e` is verified using the authorized replacement server as relay. The worker checkout stays clean at pinned `ac8e0d4`; transferring later Git objects for publication does not advance its working tree.

## In Progress

- D01: scheduler task `d18c07ab...` completed at pinned `ac8e0d4` on `owner-recovery-3080ti`. The new `mega-independent-pairclean-v2` has all five tasks; full-array diagnostics show zero exact encoded cross-split overlaps, label conflicts or structural failures.
- All four output hashes match the worker receipt; untouched tasks match source v1. Qualification remains `insufficient` pending per-task license/use boundaries, Retrieval relation-unit audit and frozen-transform learnability evidence.
- Formal experiment package remains pending data qualification and baseline selection.

## Next Actions

- 完成逐任务许可/用途边界与 Retrieval 共享文档/近重复审计；冻结变换并准备经调度执行的小规模可学习性检查
- 经已授权替代服务器同步 GitHub；继续逐任务许可边界与 Retrieval 关系单元审计

## Blockers

- The prior scheduler admission block resolved automatically; no active data job remains. D01 still lacks three qualification evidence gates listed in `docs/evidence/mega-independent-pairclean-v2-qualification.json`.
- Dataset qualification remains insufficient; historical Text JSON/checkpoint is still unavailable and is not synthesized.

## Validation

- Data pipeline: 13/13 unit tests pass under conda `QT` (Python 3.11); server data environment is Python 3.10.12 with NumPy 2.2.6.
- MEGA archive SHA-256 `b141aacd...fee13`; five-task source build is 794 MiB with full build and diagnostic manifests in `docs/evidence/`.
- Scheduler task is complete with returncode 0 and quiescent receipt; 15 split hashes and all four compact output hashes verified. Receipt: `docs/evidence/scheduler-completion-20261028.json`.
- The same seven artifact exchange tests pass on Windows and replacement-server Linux; six live HTTPS exchange checks pass.
- Linux rebuild of the portal from committed sources produces no diff; LF source hashes agree across platforms.
- Six portal routes healthy; 57 Playwright assertions pass at 1280px/390px. Registry, plan and JavaScript syntax checks pass.
- Service runtime: independent supervisor qkan-mamba-exchange; app/storage under /srv/qkan-mamba-exchange/. Secrets remain outside Git and portal.
- Local portal runs at loopback port 4191 via scripts/serve_portal.py; restart command is in research-web/README.md.

## Relevant Decisions

- User-approved collaborator execution workflow; no formal training launched.

## Plan Control

- Plan: plans/research-cycle/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: D01
- Focus Status: active
- Completed Steps: F01, F02, F03
- Next Action: 完成三项剩余资格证据；基线正式比较尚未解锁
- Blocker: None; qualification evidence remains to be completed
- Last Checkpoint: 2026-10-28
