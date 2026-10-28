# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- MEGA author-linked LRA archive is pinned and rebuilt into five adapted tasks on the replacement server; original source bytes and previous output remain preserved.
- Revision `f149439` passed 26 scheduler-run synthetic tests and is published on GitHub through the authorized server relay. The worker checkout is clean at that revision.
- New relation and train/validation-only pilots completed on the registered CPU node; no formal model training launched.

## In Progress

- D01: scheduler task `d18c07ab...` completed at pinned `ac8e0d4` on `owner-recovery-3080ti`. The new `mega-independent-pairclean-v2` has all five tasks; full-array diagnostics show zero exact encoded cross-split overlaps, label conflicts or structural failures.
- Follow-up relations audit found 9958 train/val and 9684 train/test shared encoded documents; every validation/test pair includes a document seen in training. v2 requires revision for unseen-document claims despite zero pair overlap.
- Preparing separate `mega-document-disjoint-v3`: deterministic document/verified-near-duplicate group split, cross-partition pairs discarded, all parent bytes and row provenance preserved. Not built or verified yet.
- Pilot finds signal for Text/Image; ListOps input-only parser has 0/1024 mismatches. Weak linear references for Retrieval/Pathfinder are inconclusive; they do not reject task learnability.
- Formal experiment package remains pending data qualification and baseline selection.

## Next Actions

- 经统一调度验证文档组隔离版本的构建、类分布与关系审计

## Blockers

- D01 remains gated by document isolation, appropriate learnability evidence and per-task use boundaries. LRA explicitly disallows redistributing AAN text; do not upload AAN-derived arrays to the exchange.
- Dataset qualification remains insufficient; historical Text JSON/checkpoint is still unavailable and is not synthesized.

## Validation

- At `f149439`, 26/26 tests pass through the shared scheduler (13 data + 6 qualification + 7 exchange). v3 edits remain pending another registered preflight.
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
