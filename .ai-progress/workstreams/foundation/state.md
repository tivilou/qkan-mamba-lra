# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- MEGA author-linked LRA archive is pinned and rebuilt into five adapted tasks on the replacement server; original source bytes and previous output remain preserved.
- Revision `6fd9392` passed 27 scheduler-run synthetic tests and completed the v3 data repair/audit; worker checkout is clean at that pinned revision. GitHub publication through `32bc988` is verified using separate delivery refs; worker HEAD is unchanged.
- New relation and train/validation-only pilots completed on the registered CPU node; no formal model training launched.

## In Progress

- Current bounded continuation adds exhaustive Image/Pathfinder pixel+label comparison with the pinned author archive and all-256-value reversible transform checks. Source audit discovered MEGA Pathfinder scales the common transform by 2; this is an explicit adapter difference, not a matched training recipe. New audit code is pending scheduler preflight and source-equivalence execution.
- Retrieved MEGA paper, pixel loader, LRA generators and Pathfinder generator terms; IMDb author-page recovery remains unavailable. No qualification promotion or formal model training yet.
- D01: scheduler task `d18c07ab...` completed at pinned `ac8e0d4` on `owner-recovery-3080ti`. The new `mega-independent-pairclean-v2` has all five tasks; full-array diagnostics show zero exact encoded cross-split overlaps, label conflicts or structural failures.
- Follow-up relations audit found 9958 train/val and 9684 train/test shared encoded documents; every validation/test pair includes a document seen in training. v2 requires revision for unseen-document claims despite zero pair overlap.
- `mega-document-disjoint-v3` is built and audited: Retrieval counts 117790/1804/1724; zero exact shared documents or discovered near-duplicate cross-split edges; other four task NPZ hashes unchanged from v2. All six output hashes match terminal receipt.
- Pilot finds signal for Text/Image/v3 Retrieval (58.20% versus 48.34% majority/null); ListOps input-only parser has 0/1024 mismatches. Weak Pathfinder reference is inconclusive and does not reject task learnability.
- Current assessment: docs/evidence/mega-document-v3-qualification.json, status insufficient; formal_baseline_allowed=false.
- Formal experiment package remains pending data qualification and baseline selection.

## Next Actions

- 补足 Pathfinder 的适当参考或严格源等价证据，并闭合逐任务用途记录后审定 v3 资格

## Blockers

- D01 remains gated by adequate Pathfinder evidence and per-task use records. LRA explicitly disallows redistributing AAN text; do not upload AAN-derived arrays to the exchange. Approximate deduplication does not certify author independence.
- Dataset qualification remains insufficient; historical Text JSON/checkpoint is still unavailable and is not synthesized.

## Validation

- At `6fd9392`, 27/27 tests pass through the shared scheduler (13 data + 7 qualification + 7 exchange). v3 task `87b98cf7...` complete, returncode 0, quiescent; six report hashes verified.
- All QKAN queue tasks are terminal; no running or unknown attempt remains. Existing qattn-cfra Python 3.10 environment has torch-2.7.1 package metadata, but it has not been imported/accepted for QKAN diagnostics; data jobs used the unchanged NumPy data environment.
- MEGA archive SHA-256 `b141aacd...fee13`; five-task source build is 794 MiB with full build and diagnostic manifests in `docs/evidence/`.
- Scheduler task is complete with returncode 0 and quiescent receipt; 15 split hashes and all four compact output hashes verified. Receipt: `docs/evidence/scheduler-completion-20261028.json`.
- The same seven artifact exchange tests pass on Windows and replacement-server Linux; six live HTTPS exchange checks pass.
- Linux rebuild of the portal from committed sources produces no diff; LF source hashes agree across platforms.
- Six portal routes healthy; 65 Playwright checks pass at 1280px/390px, including new v3, recent-paper and delivery projections. Registry and plan/state reconciliation pass; existing portal process retained.
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
- Next Action: 补足 Pathfinder 的适当参考或严格源等价证据，并闭合逐任务用途记录后审定 v3 资格
- Blocker: None; v3 qualification remains insufficient
- Last Checkpoint: 2026-10-28
