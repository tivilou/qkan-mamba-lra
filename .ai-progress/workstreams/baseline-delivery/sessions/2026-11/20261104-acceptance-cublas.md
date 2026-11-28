# Consolidated acceptance: CUDA environment omission

- Date: 2026-11-04
- Scope: baseline-delivery
- Inspected exchange namespace qkan-mamba/acceptance/acceptance-b5bad50719b8 (report.json and mamba1-listops-s42-uninterrupted.log), corresponding to collaborator screenshot/revision 02c3700.
- All three adapter probes passed. First Mamba/ListOps child failed before recovery comparison because deterministic matrix multiplication required CUBLAS_WORKSPACE_CONFIG. Standalone acceptance configured probes, but canary relied on the outer suite entry point for this variable.
- runner.canary now establishes :4096:8 for CUDA children. acceptance captures and automatically uploads its parent exception traceback as acceptance-error.log. Recovery tolerance and release gates unchanged.
- Verification: AST parsing of both edited modules and git diff --check passed. No model tests/training run, as requested by user. Actual rerun pending.
- Next: same consolidated collaborator command after pulling repair; automatic result review. Dataset qualification/cohort/fidelity remain formal-launch blockers.
- Compressed state.md into current snapshot; historical details remain in existing sessions and evidence files. Git baseline clean at 02c3700; changes owned by this task only.
