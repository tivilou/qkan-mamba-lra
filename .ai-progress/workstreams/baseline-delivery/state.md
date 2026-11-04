# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned baseline runners with full-state recovery, trace evidence and automatic return. Collaborators execute formal training.

## Current State

- Formal release remains blocked; actual gaps are documented in docs/baselines/release-audit-20261104.md. This is not a documentation-only closeout.
- Source audit found Mamba A_log/D _no_weight_decay flags were ignored by our optimizer. Corrected grouping preserves them; acceptance independently checks the expected parameter positions, markers and zero decay, preserves metadata across deepcopy, checks installed mamba_simple.py against retrieved pinned source hash, and binds fresh canaries to current protocol hash. No runtime receipt for this correction; do not transfer older acceptance to current code.
- Added three schema-validated candidate manifests and per-model mechanism/adaptation contracts under experiments/baseline_delivery/manifests. All remain not_verifiable, with explicit missing source-fixture/identity/runtime gates; no certificate generated.
- cohort-review.json records a proposal: Mamba1 backbone and classic Transformer; minimal S4D auxiliary only; MEGA/S5 strong-reference targets not yet integrated. B2S6 author artifact remains unresolved; LRU source unofficial; recent alternatives retained with reasons. Not a global top-x claim and not frozen. Dataset qualification must precede new integrations.
- AAN author research-only/no-redistribution/citation/publication-notice terms recovered; no author form submitted or registration inferred. CIFAR author page re-fetched, IMDb current-author 404/Stanford timeout. Foundation owns use-boundary evidence; dataset remains insufficient, no unified open license requirement invented.
- Historical acceptance acceptance-bacaa36685de at 7c05389 passed three adapter composition probes, Mamba/ListOps numerical recovery and Transformer/Retrieval exact recovery; both six-update canaries passed trace semantic/replay. Evidence: docs/evidence/baseline-acceptance-received-20261104.json. These were bounded checks at the old optimizer/protocol, not formal fidelity.
- Mamba1 v2.2.4 commit 95d8aba8a8c75aedcaa6143713b11e745e7cd0d9 remains active. Recovery tolerance mamba1-cuda-fp32-v1 is an engineering policy, not a statistical bound; CPU/other models retain exact requirements.

## Next Actions

- Resolve exact remaining task-specific internal research-use/acquisition evidence; data qualification is the current external evidence dependency. Then freeze cohort and integrate/certify selected strong baselines; combine remaining actual-device checks in one handoff rather than repeated small commands.
- Do not ask collaborator to rerun old acceptance now. Protocol hash changed: never resume old suites/checkpoints under the corrected optimizer; future device evidence must bind the corrected implementation.

## Blockers

- Dataset use/acquisition evidence still incomplete; no new-model integration before qualified data per fidelity workflow.
- MEGA/S5 strong references unimplemented, formal cohort unfrozen, current candidate manifests not_verifiable. Corrected Mamba optimizer has no runtime receipt. release.json not_released.
- User stopped owner model tests/training; do not resume. Use existing authenticated automatic artifact return for every future collaborator script.

## Validation

- Current changes: AST parsing, three manifest validators, JSON/hash/contract-link inspection and git diff --check only. No model import/test/training, package installation or scheduler job.
- Historical five-pair recovery evidence and source/install/data receipts remain in docs/evidence and prior sessions. Do not overwrite original receipts or infer all-task/long-horizon guarantees.

## Relevant Decisions

- No formal release decision. Current outcome: sessions/2026-11/20261104-release-source-audit.md.

## Plan Control

- Plan: plans/one-click/v001.json; F01 remains blocked. Current audit supersedes historical pending GPU-check wording; two bounded old-revision GPU checks have passed, current research release gaps remain above.
