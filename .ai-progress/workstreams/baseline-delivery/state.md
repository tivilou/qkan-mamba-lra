# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 68 wheels, exact MEGA/S5 source archives and CPython headers, 3,939,622,876 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Actual GPU admission PASSED at 31639c0, runs/formal-ready-20261105T042214Z-1e90315e. Authenticated gate report/evidence uploaded 2026-11-05 04:55 UTC, SHA-256 verified. All five source probes and 25 model/task recovery+trace gates passed; all five S5 agreements exact. Local certificates source_faithful_adapted. See docs/evidence/formal-readiness-passed-20261105.json.
- Earlier screenshot confirmed Transformer/ListOps Epoch4/100, step11705/300000. Latest screenshot after user-requested restart shows zero running, one paused,24 pending; current training continuation is NOT confirmed. Latest three exchange failures (8cdee9a4a20a,bc519091a5f0,f61ee663a5cc) are KeyboardInterrupt at31639c0, not evidence of a new model error. Original readiness repeats environment/data checks before updating saved task states. Independent display now distinguishes watch-only, parent liveness, startup elapsed time and log activity; new launches restrict stage markers to current-attempt log offsets. Executor stays31639c0. See docs/baselines/formal-progress.md.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Runtime success is established only for the recorded revision/device/protocol; no completed formal training result is established.

## Next Actions

- Preserve the current window; inspect progress-parent.log and setup.log in another tmux window to establish this attempt's actual stage. Watch-only cannot resume paused jobs. Fetch/extract the independent monitor without pulling the executor; do not launch a duplicate parent or repeatedly interrupt readiness checks. A fresh running job/worker heartbeat must establish resumed training.
- Collect automatically returned formal results and distinguish admission from job completion/performance.

## Blockers

- No admission blocker for recorded runtime. Current pause/preparation vs actual launch mode needs collaborator parent-log observation; saved paused counts alone cannot distinguish it. No owner access to collaborator live process.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Existing admission evidence unchanged. Prior display/rate/renderer checks retained. Stdlib AST and temporary startup fixtures cover paused watch-only, live preparation, current-attempt stage/data markers, historical marker exclusion and stale/dead parent heartbeat. No framework/model import, install, training or live interruption. Worker phase granularity remains limited to existing heartbeat fields.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 complete for the recorded runtime; ongoing formal-result follow-up remains.
