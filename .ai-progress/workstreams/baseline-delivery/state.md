# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 68 wheels, exact MEGA/S5 source archives and CPython headers, 3,939,622,876 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Actual GPU admission PASSED at 31639c0, runs/formal-ready-20261105T042214Z-1e90315e. Authenticated gate report/evidence uploaded 2026-11-05 04:55 UTC, SHA-256 verified. All five source probes and 25 model/task recovery+trace gates passed; all five S5 agreements exact. Local certificates source_faithful_adapted. See docs/evidence/formal-readiness-passed-20261105.json.
- Latest user screenshot confirms resumed Transformer/ListOps Epoch10/100, step27101/300000 (9.03%), running1/pending24/failed0, PID46829, GPU96%,0.79/24GiB. Executor stays31639c0. GPU occupancy and startup/mode visibility are in the independent monitor. See docs/baselines/formal-progress.md.
- User authorized adaptive throughput refinement. Both PyTorch and native S5/JAX retain physical tiers1/2/4/8/16/32 and logical batch32/FP32. Removed the5% gain floor: positive gains require at least3 confirmation windows; one slow window does not revert. Sustained confirmed slowdown or OOM can lower the tier; mixed/equal trials retain the candidate after8 windows without claiming speedup. Controller and partial confirmation windows persist in full-state checkpoints.
- Implemented full logical-batch preparation/staging, one-batch CPU prefetch, per-logical-update device statistics, five-second advisory writes with transition/pause flush, and reusable GPU rollback buffers with checkpoint-owned CPU fallback (immutable candidate state for S5). Prefetch does not advance the committed sampler. This remains a distinct adaptive numerical trajectory, not a migration of the current fixed-policy suite. See docs/baselines/adaptive-memory.md and docs/evidence/adaptive-memory-throughput-20261105.json.
- New scripts/run_adaptive_readiness.sh enters source/recovery/trace gates for a NEW adaptive suite. Code-only prepare_adaptive_checkout.sh creates a separate revision-pinned sibling worktree and reuses local assets without touching active tracked code or starting GPU work. Candidate manifests/policy hashes refreshed; release remains closed and adaptive hardware performance/recovery remain unverified.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Runtime success is established only for the recorded revision/device/protocol; no completed formal training result is established.

## Next Actions

- Preserve current fixed-policy training/checkpoints; do not pull new training code into it. When preparing adaptation, fetch/extract prepare_adaptive_checkout.sh into ignored .bootstrap and run the code-only helper. Schedule the new checkout's adaptive --gates-only entry on an available GPU; inspect returned evidence before treating it as runtime verified. Do not overlap GPU0 probes with current training merely for an upgrade.
- Collect automatically returned formal results and distinguish admission from job completion/performance.

## Blockers

- Adaptive throughput code is committed locally at d50da46. GitHub publication is pending: configured local proxy terminates TLS for github.com/api.github.com, and direct github.com:443 connection fails. Existing credentials were available; no authentication error was observed. Do not tell the collaborator to pull this revision until publication succeeds.
- No admission/startup blocker reported; screenshot confirms advancing formal training. No owner access to collaborator live process; checkpoint continuity beyond the screenshot awaits runtime evidence.
- Adaptive mechanism is code-delivered only: actual CUDA rollback, tier throughput/peaks, model recovery and Linux shell execution remain collaborator checks. No claimed speedup or equivalence to the old fixed-policy trajectory.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Existing31639c0 admission evidence unchanged. New28 stdlib/mock checks pass:1% retention, multi-window confirmation and partial restore, bounded inconclusive trials, OOM transaction/RNG rollback, finite exhaustion, independent jobs, fresh-process controller restore, specific JAX OOM classification, prefetch order/failure/cleanup, reusable buffer independence/CPU mode and status throttling. AST and candidate source/manifest/release checks pass. No owner framework/model import, GPU query, install, training or live interruption. Actual GPU behavior and speed remain pending collaborator evidence.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 complete for the recorded runtime; ongoing formal-result follow-up remains.
