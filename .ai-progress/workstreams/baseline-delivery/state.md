# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 68 wheels, exact MEGA/S5 source archives and CPython headers, 3,939,622,876 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Actual GPU admission PASSED at 31639c0, runs/formal-ready-20261105T042214Z-1e90315e. Authenticated gate report/evidence uploaded 2026-11-05 04:55 UTC, SHA-256 verified. All five source probes and 25 model/task recovery+trace gates passed; all five S5 agreements exact. Local certificates source_faithful_adapted. See docs/evidence/formal-readiness-passed-20261105.json.
- User confirms training started automatically after waiting on the zero-running/one-paused display; no further restart or diagnostic-log request is needed for that incident. This is consistent with original readiness rechecking environments/data before scheduling. Earlier screenshot recorded Transformer/ListOps Epoch4/100, step11705/300000; current resumed step is not independently observed. Independent display patch3a2848e distinguishes watch-only, parent liveness, startup elapsed time and log activity; new launches restrict stage markers to current-attempt log offsets. Executor stays31639c0. See docs/baselines/formal-progress.md.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Runtime success is established only for the recorded revision/device/protocol; no completed formal training result is established.

## Next Actions

- Leave the resumed training running. No restart/pull is needed for display patch3a2848e; use fetch-only watch-only extraction in a separate tmux window if observation upgrades are needed. Review later returned formal artifacts for completion and continuation evidence.
- Collect automatically returned formal results and distinguish admission from job completion/performance.

## Blockers

- No admission/startup blocker reported: user confirms automatic training startup after waiting. No owner access to collaborator live process; current optimizer step and checkpoint continuity await runtime evidence.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Existing admission evidence unchanged. Prior display/rate/renderer checks retained. Stdlib AST and temporary startup fixtures cover paused watch-only, live preparation, current-attempt stage/data markers, historical marker exclusion and stale/dead parent heartbeat. No framework/model import, install, training or live interruption. Worker phase granularity remains limited to existing heartbeat fields.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 complete for the recorded runtime; ongoing formal-result follow-up remains.
