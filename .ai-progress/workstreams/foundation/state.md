# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- MEGA author-linked LRA archive is pinned and rebuilt into five adapted tasks on the replacement server; original source bytes and previous output remain preserved.
- Worker checkout is clean at `52257f2`: 31 scheduler-run tests pass; exhaustive Image/Pathfinder source-equivalence audit complete. Prior GitHub checkpoint `19b7d8e` is verified; this continuation's publication is pending.
- New relation and train/validation-only pilots completed on the registered CPU node; no formal model training launched.

## In Progress

- D01 blocked on per-task use evidence. Pathfinder source/learnability gate now passes via exact author-package equivalence, all-256-value invertibility and the primary MEGA reference; no local trained Pathfinder result is claimed.
- Source audit found MEGA Pathfinder scales the common input by 2; Image mean/std also differ. These require explicit baseline adapter/fidelity evidence, not silent preprocessing changes. See docs/pathfinder-source-audit.md.
- Retrieved MEGA paper, pixel loader, LRA generators, Pathfinder MIT code license and Stanford IMDb dataset card. Author-page recovery failed; card says other/More Information Needed for licensing. No qualification promotion.
- v2 pair audit missed shared documents in every validation/test pair; preserved v2 is not eligible for unseen-document claims.
- `mega-document-disjoint-v3` is built and audited: Retrieval counts 117790/1804/1724; zero exact shared documents or discovered near-duplicate cross-split edges; other four task NPZ hashes unchanged from v2. All six output hashes match terminal receipt.
- Pilot finds signal for Text/Image/v3 Retrieval (58.20% versus 48.34% majority/null); ListOps input-only parser has 0/1024 mismatches. Diagnostics remain outside the formal baseline table.
- Current assessment: docs/evidence/mega-document-v3-qualification.json, status insufficient; formal_baseline_allowed=false.
- Formal experiment package remains pending data qualification and baseline selection.

## Next Actions

- 取得逐任务权威用途与获取声明，闭合剩余数据资格门槛

## Blockers

- Per-task use evidence remains incomplete; author-page connection failures and the IMDb card do not establish permission terms. LRA disallows redistributing AAN text; no dataset arrays/raw text are currently delivered through our exchange. Approximate deduplication does not certify author independence.
- Dataset qualification remains insufficient; historical Text JSON/checkpoint is still unavailable and is not synthesized.

## Validation

- At `52257f2`, 31/31 tests pass through the shared scheduler (13 data + 7 qualification + 4 pixel source + 7 exchange). Tasks `913b7ae8...` and `cc37d99e...` complete, returncode 0, quiescent; both downloaded report hashes verified. All 260000 Image/Pathfinder rows exactly match source pixels and labels.
- All QKAN queue tasks are terminal; no running or unknown attempt remains. Existing qattn-cfra Python 3.10 environment has torch-2.7.1 package metadata, but it has not been imported/accepted for QKAN diagnostics; data jobs used the unchanged NumPy data environment.
- MEGA archive SHA-256 `b141aacd...fee13`; five-task source build is 794 MiB with full build and diagnostic manifests in `docs/evidence/`.
- Scheduler task is complete with returncode 0 and quiescent receipt; 15 split hashes and all four compact output hashes verified. Receipt: `docs/evidence/scheduler-completion-20261028.json`.
- The same seven artifact exchange tests pass on Windows and replacement-server Linux; six live HTTPS exchange checks pass.
- Linux rebuild of the portal from committed sources produces no diff; LF source hashes agree across platforms.
- Six portal workbenches and seven HTTP routes healthy; current source/use projections pass 69 browser checks at 1280px/390px. Registry and plan/state reconciliation pass; original portal PID 67612 retained. Receipt: docs/evidence/portal-pixel-refresh-20261028.json.
- Service runtime: independent supervisor qkan-mamba-exchange; app/storage under /srv/qkan-mamba-exchange/. Secrets remain outside Git and portal.
- Local portal runs at loopback port 4191 via scripts/serve_portal.py; restart command is in research-web/README.md.

## Relevant Decisions

- User-approved collaborator execution workflow; no formal training launched.

## Plan Control

- Plan: plans/research-cycle/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: D01
- Focus Status: blocked
- Completed Steps: F01, F02, F03
- Next Action: 取得逐任务权威用途与获取声明，闭合剩余数据资格门槛
- Blocker: Per-task authoritative use/acquisition evidence incomplete; see docs/data-use-boundaries.md
- Last Checkpoint: 2026-10-28
