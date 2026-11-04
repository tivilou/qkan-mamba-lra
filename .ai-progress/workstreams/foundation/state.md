# 研究恢复、协作交付与科研门户

## Goal

- Restore code provenance, collaborator delivery and the local research portal; prepare the next data/literature stage without formal training.

## Current State

- 2026-11-04: D01 completed. Original IMDb paper recovered from ACL; explicit future-research dataset release plus per-task author-source/acquisition assessment closes use evidence for internal non-commercial research/no-data-redistribution only. docs/evidence/task-internal-use-review-20261104.json and imdb-primary-paper-sources-20261104.json support qualified data; model release remains false. Session: sessions/2026-11/20261104-data-qualified.md.

- GitHub 74acc3d recovered to local branch research/collaboration-foundation; original 3080Ti released, replacement explicitly selected via serverctl development role.
- F01/F02/F03 complete: code provenance restored, isolated HTTPS exchange accepted, six portal workbenches validated. sigmoidpp has accepted write access.
- Collaborator's new GitHub account `bigmi-cuc` received a write-access invitation on 2026-11-01; invitation `340846475` is pending acceptance. Prior account access was not revoked. Receipt: sessions/2026-11/20261101T084458Z-collaborator-account.md.
- MEGA author-linked LRA archive is pinned and rebuilt into five adapted tasks on the replacement server; original source bytes and previous output remain preserved.
- Worker checkout is clean at `52257f2`: 31 scheduler-run tests pass; exhaustive Image/Pathfinder source-equivalence audit complete. GitHub publication through `2600b4a` is verified via separate delivery refs, and draft PR #1 description reflects the current scope.
- New relation and train/validation-only pilots completed on the registered CPU node; no formal model training launched.

## In Progress

- Earlier same-day AAN/CIFAR source recovery and IMDb website failures are preserved in prior receipts. The original author paper now supplies research-release evidence; see the current qualified assessment above.
- D01 is now done for the declared internal research scope. Pathfinder source/learnability gate now passes via exact author-package equivalence, all-256-value invertibility and the primary MEGA reference; no local trained Pathfinder result is claimed.
- Source audit found MEGA Pathfinder scales the common input by 2; Image mean/std also differ. These require explicit baseline adapter/fidelity evidence, not silent preprocessing changes. See docs/pathfinder-source-audit.md.
- Retrieved MEGA paper, pixel loader, LRA generators, Pathfinder MIT code license and Stanford IMDb dataset card. Author-page recovery failed; card says other/More Information Needed for licensing. No qualification promotion.
- v2 pair audit missed shared documents in every validation/test pair; preserved v2 is not eligible for unseen-document claims.
- `mega-document-disjoint-v3` is built and audited: Retrieval counts 117790/1804/1724; zero exact shared documents or discovered near-duplicate cross-split edges; other four task NPZ hashes unchanged from v2. All six output hashes match terminal receipt.
- Pilot finds signal for Text/Image/v3 Retrieval (58.20% versus 48.34% majority/null); ListOps input-only parser has 0/1024 mismatches. Diagnostics remain outside the formal baseline table.
- Current data assessment: docs/evidence/mega-document-v3-qualification.json is qualified, formal_baseline_allowed=true within documented scope. release.json still not_released; no formal training authorized by this data flag alone.
- Historical orchestrator tracking is not promoted from its old blocked snapshot automatically; current project plan has moved D01 to done and R01 active. Baseline release remains pending.
- Native MEGA/S5 workers/environment recipes/recovery/trace and a conditional one-command launch are delivered in baseline-delivery. Current-revision runtime evidence remains pending; the collaborator entry runs all gates before automatic formal admission.

## Next Actions

- Complete native MEGA/S5 training, recovery and trace adapters from verified official source factories.

## Blockers

- No current data-use evidence blocker within the reviewed internal scope. No raw/derived data redistribution; approximate deduplication still does not certify exhaustive semantic or author independence.
- Native baseline implementation/environment/certification remains incomplete. Historical Text JSON/checkpoint remains unavailable; do not synthesize it.

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
- Focus Step: R01
- Focus Status: active
- Completed Steps: F01, F02, F03, D01
- Next Action: Collaborator executes consolidated launch; audit returned all-gate evidence and resulting formal suite
- Blocker: Native delivery complete; actual-device source/recovery/trace receipts remain pending.
- Last Checkpoint: 2026-11-04
