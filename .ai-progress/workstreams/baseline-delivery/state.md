# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 68 wheels, exact MEGA/S5 source archives and CPython headers, 3,939,622,876 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Actual GPU admission PASSED at 31639c0, runs/formal-ready-20261105T042214Z-1e90315e. Authenticated gate report/evidence uploaded 2026-11-05 04:55 UTC, SHA-256 verified. All five source probes and 25 model/task recovery+trace gates passed; all five S5 agreements exact. Local certificates source_faithful_adapted. See docs/evidence/formal-readiness-passed-20261105.json.
- Latest screenshot confirms formal scheduler running one job with 24 pending and no reported failure. User authorized display upgrade and restart via checkpoint continuation. Independent scripts/formal_progress.py supervisor now adds 15s readable snapshots without changing executor revision31639c0 or runtime admission. Fetch/extract only this file into ignored .bootstrap; never git pull that active executor checkout. See docs/baselines/formal-progress.md.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Runtime success is established only for the recorded revision/device/protocol; no completed formal training result is established.

## Next Actions

- In qkan tmux, safely pause old runner once and wait for prompt, then fetch/extract new display supervisor and run it with --resume runs/formal-ready-20261105T042214Z-1e90315e. Supervisor checks unchanged executor revision, clean tracked sources and free suite lock, then calls original entry with original cache/GPU. Completed jobs retained; unfinished jobs use latest valid checkpoints. No new suite.
- Collect automatically returned formal results and distinguish admission from job completion/performance.

## Blockers

- No admission blocker for recorded runtime. Display-supervised Linux continuation needs collaborator execution. Original runner grants90s safe-pause grace; if exceeded, latest committed checkpoint is retained but work after it may repeat. No owner access to collaborator live process.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Existing admission evidence unchanged. New supervisor: stdlib AST/help and temporary fixtures verify counts, epoch/batch derivation, measured rate/ETA, stale/old-process heartbeat handling. No model import, install or training; no live process interrupted. Worker phase granularity remains limited to existing heartbeat fields; supervisor reports that explicitly.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 complete for the recorded runtime; ongoing formal-result follow-up remains.
