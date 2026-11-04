# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 65 wheels, exact MEGA/S5 source archives and CPython headers, 3,929,680,447 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Latest actual attempt at 33046c4, runs/formal-ready-20261104T125528Z-e67c68dd: all MEGA/S5 isolated setup completed (s5:verify, pip check clean), data content verification passed, execution stopped at MEGA source forward/gradient/update probe. Failure-6152ebac4e46 retrieved. The source-probe log is 3903 bytes, SHA256 0db0f9ceed2466cad9fc380d9da2f9fa6e929fc519fae0058ff22cf6b20488e9; only its hash was returned. Its traceback is needed before a mechanism/dependency fix; do not guess the cause or rerun training. Python.h repair is confirmed past compilation. Formal training has not started.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Current GPU/runtime success is not established.

## Next Actions

- Collaborator pulls this complete delivery and runs bash scripts/run_formal_readiness.sh --cache "$QKAN_CACHE" --gpus 0 in the existing configured terminal. Use a fresh suite, never --resume a previous code revision. See docs/baselines/consolidated-launch.md.
- Inspect actual returned gates/results. Source presence and historical acceptance do not certify the current implementation/protocol.

## Blockers

- MEGA source probe failed; specific exception pending the existing mega-source-probe.log. Remaining source/recovery/trace gates have not passed. Next: inspect that existing log, fix the evidenced cause, and include bounded synthetic source-probe diagnostics in future automatic returns.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Owner staging: dependency resolution/download and pure-Python ANTLR wheel preparation without installation. Actual asset publication/hash/readback checks; Python AST, manifest schema/provenance/source pins, requirements and reviewed policy hashes. No owner model execution.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 remains blocked on actual runtime evidence.
