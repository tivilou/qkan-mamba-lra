# Baseline delivery

## Goal

- Deliver source-pinned, recoverable baseline experiments for collaborator execution with automatic artifact return.

## Current State

- First consolidated execution at dd4c1d2, runs/formal-ready-20261104T112338Z-ddafd722, failed before training: system Python3.10 lacks ensurepip. Received failure-5f53160eebb4 plus collaborator screenshot identify the cause. Fixed complete-Python selection, isolated Conda fallback and preserved incomplete-env recreation. New compact setup diagnostics return automatically. Current fix has no actual-device receipt.

- Native MEGA and S5 task/training/checkpoint/trace workers and isolated environment recipes are delivered. S5 retains native JAX/Flax source scan and Optax standard groups. MEGA uses official fairseq encoder and clears derived EMA eval caches on weight loading. See docs/baselines/consolidated-launch.md.
- One command runs isolated setup, source forward/loss/gradient/update probes, all five-model/five-task fresh-process recovery and complete trace gates, returns compact evidence, then automatically starts the 25 full-data seed42 jobs only if all gates pass. Per-device evidence and implementation/protocol/source/policy hashes bind a local release. Committed release remains not_released and all five manifests not_verifiable until runtime evidence exists.
- Dataset is qualified for documented internal noncommercial/no-data-redistribution scope. Main roster Transformer/Mamba1/MEGA/S5; minimal S4D auxiliary. No author benchmark reproduction, SOTA/top-two or quantum claim.
- Mamba corrected no-decay optimizer still needs current runtime evidence; historical acceptance-bacaa36685de at 7c05389 is not transferred. New entry uses a fresh suite and protocol; old suite cannot resume here.

## Next Actions

- Collaborator pulls the repair and runs the same consolidated launch command in a new suite (no --resume to the old revision). Successful checks automatically start formal training and return completion artifacts; failed checks stop with compact return receipt and local logs. Owner must inspect actual returned evidence before claiming readiness/results.

## Blockers

- The dd4c1d2 setup defect is evidenced and repaired in code; current repair/environment, source probes, exact recovery and all-task trace still need collaborator execution. No formal training has started.
- User stopped owner model tests/training/installations; continue to honor this instruction.

## Validation

- Python AST/JSON, manifest and artifact hashes, policy bindings, source/API inspection, pinned package metadata and shell syntax only. No model imports/tests, installations or experiment launched on owner hardware.

## Plan Control

- plans/one-click/v001.json F01 remains blocked on current runtime evidence, not missing native worker code. Current entry scripts/run_formal_readiness.sh; docs/baselines/consolidated-launch.md.
