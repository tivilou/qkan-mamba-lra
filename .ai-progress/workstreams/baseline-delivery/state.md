# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 68 wheels, exact MEGA/S5 source archives and CPython headers, 3,939,622,876 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Latest actual attempt at 22dc743, runs/formal-ready-20261104T134648Z-d8ac91ae: setup/data and all five source probes completed; control flow reached S5/ListOps after Transformer/Mamba1/MEGA all-task canaries. Retrieved failure-80f2fef49ca9 and supplied traceback establish first S5 uninterrupted child failed at logit trace replay (max abs 0.00055149, rtol1e-5/atol1e-6), before S5 resume comparison. Forward used JIT Dense versus eager replay with unspecified dot precision. Explicit HIGHEST head precision and compiled independent replay are delivered; source encoder and tolerances preserved. Actual-device repair unverified; formal training remains unstarted. See docs/evidence/s5-trace-replay-repair-20261104.json.
- Failures now return worker exception types/project frame locations and aggregate replay differences automatically, alongside existing setup/synthetic source-probe diagnostics. Real sample arrays, exception text/locals and task logs remain private/local.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Current GPU/runtime success is not established.

## Next Actions

- Collaborator pulls this complete delivery and runs bash scripts/run_formal_readiness.sh --cache "$QKAN_CACHE" --gpus 0 in the existing configured terminal. Use a fresh suite, never --resume a previous code revision. See docs/baselines/consolidated-launch.md.
- Inspect actual returned gates/results. Source presence and historical acceptance do not certify the current implementation/protocol.

## Blockers

- Need collaborator GPU receipt for the revised S5 head/replay and remaining all-task recovery/trace gates. No additional manual log extraction or dependency command is required. Same consolidated entry launches formal jobs automatically only after every gate and verified evidence return succeeds.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Owner staging: dependency resolution/download and pure-Python ANTLR wheel preparation without installation. Actual asset publication/hash/readback checks; Python AST, manifest schema/provenance/source pins, requirements and reviewed policy hashes. No owner model execution.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 remains blocked on actual runtime evidence.
