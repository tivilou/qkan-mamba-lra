# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned baseline runners with full-state recovery, trace evidence and automatic artifact return. Collaborators run formal training.

## Current State

- Latest collaborator acceptance `acceptance-bacaa36685de` at `7c05389` is engineering_checks_complete. Direct exchange review confirms all three adapter probes passed, Mamba/ListOps recovery passed mamba1-cuda-fp32-v1, Transformer/Retrieval recovery exact, both six-update fresh-process canaries with trace semantic/replay passed. Formal release remains false. Evidence: docs/evidence/baseline-acceptance-received-20261104.json. This supersedes the pending rerun below.
- Collaborator acceptance `acceptance-b5bad50719b8` at revision `02c3700` failed. Exchange report and uninterrupted worker log inspected directly on 2026-11-04. Transformer, S4D minimal and Mamba1 adapter composition probes passed; these are not paper-level fidelity certificates.
- First Mamba/ListOps worker exited at its initial matrix multiplication: deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG, omitted by the standalone acceptance entry point. Recovery comparison and Transformer/Retrieval canary were not reached.
- Fixed runner.canary to set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting CUDA children, covering standalone callers. Acceptance now automatically returns its full parent exception traceback as acceptance-error.log. Collaborator rerun pending; no owner model tests/training performed per user instruction. AST syntax and diff checks pass.
- Consolidated command: docs/baselines/consolidated-acceptance.md. Uses original suite `runs/baseline-20261103T115633Z-fa88b308`, fresh acceptance directories, three adapter probes and Mamba/ListOps plus Transformer/Retrieval recovery/trace canaries. Required diagnostic artifacts automatically upload to `qkan-mamba/acceptance/<id>` with verified receipts.
- Mamba1 CUDA FP32 recovery policy `mamba1-cuda-fp32-v1`: symmetric per-element rtol=1e-5, atol=1e-6, exact controls/discrete/selection state. This is engineering tolerance, not a statistical or long-horizon guarantee. CPU/other models retain exact recovery.
- Five-pair collaborator GPU diagnostic received and verified: model maximum full/full 7.23e-7, recovery 5.74e-7; optimizer both 3.73e-9. Leaf review found 60/1311 leaves exceeding their own observed repeat maxima; maxima do not define a calibrated bound. Evidence: docs/evidence/baseline-recovery-replicates-received-20261103.json and baseline-recovery-leaf-review-20261103.json.
- Engineering candidates: Transformer, explicitly pedagogical upstream minimal S4D, pinned Mamba1 v2.2.4 commit 95d8aba8a8c75aedcaa6143713b11e745e7cd0d9. No frozen formal cohort or per-model fidelity certification.

## Next Actions

- No repeat of the same acceptance requested. Close remaining research qualification/release work on our side; two bounded CUDA canaries are now passed, not all task/model combinations.
- Complete source/use qualification, evidence-grounded cohort selection and model fidelity certificates, tracked in docs/baselines/formal-launch-closeout.md. Acceptance success alone does not release formal experiments.

## Blockers

- Formal dataset qualification remains insufficient on source/use evidence. release.json remains not_released with no frozen cohort/certificates. No formal-ready launch package or performance result.
- Owner tests stopped by user; do not resume them. Do not contact the released historical GPU endpoint. Resolve authorized development server through serverctl.

## Validation

- Latest collaborator CUDA acceptance passed as recorded above; no additional owner tests/training run.
- Prior collaborator verified all 15 splits by internal NPY content and installed mirrored Torch 2.7.1+cu126/Mamba 2.2.4. Trace replay GPU fix remains subject to consolidated acceptance.
- Historical validation receipts and implementation sessions remain in docs/evidence and sessions/2026-11; no old CPU check certifies present GPU acceptance.

## Relevant Decisions

- No formal release decision. Session: sessions/2026-11/20261104-acceptance-cublas.md.

## Plan Control

- Plan: plans/one-click/v001.json; focus F01 blocked. Formal release requires data qualification, cohort/fidelity and collaborator GPU acceptance.
