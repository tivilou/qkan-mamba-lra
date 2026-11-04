# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- Offline assets published: 65 wheels, exact MEGA/S5 source archives and CPython headers, 3,929,680,447 bytes. All upload receipts, stored chunks and reconstructed hashes verified; both source archives and pip bootstrap read back over authenticated HTTPS. See strong-offline-mirror.json and docs/evidence/strong-offline-publication-20261104.json.
- User requires no external network on collaborator. Setup uses existing local Python3.10/stdlib venv --without-pip, verified local pip23.3.2 bootstrap, exchange-only wheel/source staging and --no-index installs/build. No Conda/apt/public pip/Git/author-data fallback. Qualified local task cache is reused and verified. S5 NVRTC is pinned to CUDA12.2 alongside its runtime.
- Latest actual attempt at 55c8abf, runs/formal-ready-20261104T122816Z-a6915f78: offline MEGA dependencies and local Python bootstrap succeeded, then compilation failed because system Python.h is absent. Retrieved failure-d7c52bb1f892 and private redacted setup log establish the cause. Published/read back a 240896-byte, 164-file unmodified Ubuntu CPython3.10 amd64 header closure; native build now includes it from project-local cache without apt/system mutation. Current runtime repair remains unverified. See docs/evidence/python-headers-repair-20261104.json.
- Native MEGA/S5 workers remain source-faithful task adapters. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. Dataset qualified for documented internal noncommercial/no-data-redistribution scope. No author benchmark reproduction, SOTA/top-two or quantum claim.
- One entry installs, runs source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and trace gates, returns compact evidence, then automatically starts 25 full-data seed42 jobs only when every gate passes. Committed release stays not_released and all five manifests not_verifiable. Current GPU/runtime success is not established.

## Next Actions

- Collaborator pulls this complete delivery and runs bash scripts/run_formal_readiness.sh --cache "$QKAN_CACHE" --gpus 0 in the existing configured terminal. Use a fresh suite, never --resume a previous code revision. See docs/baselines/consolidated-launch.md.
- Inspect actual returned gates/results. Source presence and historical acceptance do not certify the current implementation/protocol.

## Blockers

- Actual collaborator offline environment build, source probes, recovery and trace gates pending; no formal training started. Local Python3.10/stdlib venv and native build tools are prerequisites. Missing assets stop without internet fallback.
- No owner model imports/tests, dependency installation or experiment runs, per user instruction.

## Validation

- Owner staging: dependency resolution/download and pure-Python ANTLR wheel preparation without installation. Actual asset publication/hash/readback checks; Python AST, manifest schema/provenance/source pins, requirements and reviewed policy hashes. No owner model execution.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 remains blocked on actual runtime evidence.
