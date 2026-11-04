# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned baseline runners with full-state recovery, trace evidence and automatic return. Collaborators execute formal training.

## Current State

- Data qualification is now qualified for documented internal non-commercial/no-data-redistribution scope. Implementation roster fixed: Transformer/Mamba1/MEGA/S5 main, minimal S4D auxiliary. Formal release remains blocked on actual native integration/certification, not IMDb evidence. Current source-stage status: docs/baselines/strong-source-integration.md.
- Source audit found Mamba A_log/D _no_weight_decay flags were ignored by our optimizer. Corrected grouping preserves them; acceptance independently checks the expected parameter positions, markers and zero decay, preserves metadata across deepcopy, checks installed mamba_simple.py against retrieved pinned source hash, and binds fresh canaries to current protocol hash. No runtime receipt for this correction; do not transfer older acceptance to current code.
- Added three schema-validated candidate manifests and per-model mechanism/adaptation contracts under experiments/baseline_delivery/manifests. All remain not_verifiable, with explicit missing source-fixture/identity/runtime gates; no certificate generated.
- cohort-review.json has selection_frozen=true but formal_cohort_frozen=false. strong_sources.py verifies official clean repository/commit/file/import identities and constructs actual MegaLRAEncoder and Flax S5 encoder. strong-source-registry.json pins APIs/dependencies. This is source-factory-only: no native training worker, environment, full-state recovery or trace is certified. New MEGA/S5 manifests remain not_verifiable. B2S6 and other untested candidates/claim limits remain explicit.
- Foundation closed task-use evidence through original IMDb author paper and per-task source assessment; no author website success, AAN registration, blanket license or data redistribution right inferred. Dataset bytes/splits/claim limits unchanged. S5 upstream GPU requirements contain unresolved jax[cuda]>=version, so a reproducible compatible environment remains real work.
- Historical acceptance acceptance-bacaa36685de at 7c05389 passed three adapter composition probes, Mamba/ListOps numerical recovery and Transformer/Retrieval exact recovery; both six-update canaries passed trace semantic/replay. Evidence: docs/evidence/baseline-acceptance-received-20261104.json. These were bounded checks at the old optimizer/protocol, not formal fidelity.
- Mamba1 v2.2.4 commit 95d8aba8a8c75aedcaa6143713b11e745e7cd0d9 remains active. Recovery tolerance mamba1-cuda-fp32-v1 is an engineering policy, not a statistical bound; CPU/other models retain exact requirements.

## Next Actions

- Implement native MEGA/S5 task/head/training, checkpoint and sample-trace adapters, with isolated pinned environments and source-mechanism comparison. Combine remaining actual-device checks and automatic return in one handoff.
- Do not ask collaborator to rerun old acceptance now. Protocol hash changed: never resume old suites/checkpoints under the corrected optimizer; future device evidence must bind the corrected implementation.

## Blockers

- Data gate closed within its documented scope; continue native source integration. Formal release still requires executable environments, source gate evidence and actual-device checks.
- MEGA/S5 source factories written but native workers/recovery/trace unavailable; S5 GPU dependency placeholder unresolved. Five candidate manifests remain not_verifiable. Corrected Mamba optimizer has no runtime receipt. release.json not_released.
- User stopped owner model tests/training; do not resume. Use existing authenticated automatic artifact return for every future collaborator script.

## Validation

- Current changes: AST/manifest/JSON/source hashes and deterministic portal projection checks only; no model import/test/training, package installation or scheduler job. Primary-paper text read using bundled PDF runtime; other stdlib work uses existing QT2.
- Historical five-pair recovery evidence and source/install/data receipts remain in docs/evidence and prior sessions. Do not overwrite original receipts or infer all-task/long-horizon guarantees.

## Relevant Decisions

- No formal release decision. Current outcome: sessions/2026-11/20261104-strong-source-factories.md.

## Plan Control

- Plan: plans/one-click/v001.json; F01 remains blocked. Current audit supersedes historical pending GPU-check wording; two bounded old-revision GPU checks have passed, current native integration/certification gaps remain above.
