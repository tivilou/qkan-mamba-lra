# Critical Invariants: 基线复现与可验证实验交付

<!--
Keep this file short. Record only current rules that constrain edits,
publication, experiments, data access, workspace boundaries, or claims.
Put rationale and chronology in decisions/*.md instead.
-->

## Scope

- Workstream: `baseline-delivery`.

## Mandatory Rules

- We implement and verify code; collaborators run formal experiments. Owner-side diagnostics use the shared research scheduler.
- Formal comparison requires a qualified dataset, frozen evidence-grounded cohort, and per-model fidelity certification. Preparatory literature and diagnostic work do not unlock formal runs.
- Preserve historical implementations and results. The legacy `src/baselines/s4.py` is an unverified simplified model, not an official S4 reproduction.
- Token IDs require embeddings and padding masks; Retrieval keeps two document identities with a declared pair head. Test data never select transforms, hyperparameters or checkpoints.
- Long jobs require latest full-state resume independently of best-validation selection and a fresh-process recovery canary. Formal readiness also requires real sample-trace.v2 structural, semantic and replay evidence.
- Keep source provenance, reproduction fidelity, common-protocol performance and scientific claims separate; never certify a baseline from its name or import alone.
- Public Git holds code/configuration and reviewed compact evidence. Data, weights, raw traces and credentials remain outside Git and use the authenticated project exchange when shared.

- User instruction (2026-11-03): future collaborator scripts must automatically return explicitly needed runtime artifacts to the designated project exchange directory, with verified receipts and retryable local retention; reuse existing authenticated return tooling.
- User instruction (2026-11-04): assume the collaborator cannot access external internet. Runtime/install scripts use local prerequisites/data and authenticated exchange only; prepare source/dependency assets on our side, disable public-source fallbacks, and stop clearly if an asset is missing. No owner model tests, training or dependency installation.
- User instruction (2026-11-18): one recommended experiment entry; train/resume and original progress dashboard in one terminal. Use explicit watch for read-only observation. Preserve existing Suite executor revision and scientific bindings; unified control scripts may live in ignored .bootstrap without replacing training code.
- User instruction (2026-11-28, supersedes Nov25Retrieval-last priority): resume unfinished jobs shortest estimated remaining time first, using reviewed14job measurements. Use `--task-order fastest-first`; retain selection on later resume. Preserve completed results/latest checkpoints/scientific budgets. Explicit `retrieval-last` or `manifest` remains a scheduler-only rollback.

## Decision Index

- Execution architecture (2026-11-24): new accelerations use explicit worker hooks and model-instance plugins; do not replace shared trainer/upstream globals. Science binding excludes UI/controller, but every Suite retains its original pinned executor. See docs/baselines/execution-architecture.md. Migrate only from latest full state, never a historical fixed step.

<!-- Add one bullet per decision: `- [STATUS] Short title -> [detail](relative/path.md)`. -->
- [ACTIVE] Implementation roster and claim scope -> [detail](decisions/2026-11-04-implementation-roster.md)
- [ACTIVE] Assume collaborator has no external internet -> [detail](decisions/2026-11-04-offline-collaborator.md)
- [ACTIVE] Unified experiment entry with pinned executors -> [detail](decisions/2026-11-18-unified-experiment-entry.md)
- [ACTIVE] User-selected MEGA Text prospective precision amendment; other dispatch remains pinned -> [detail](decisions/2026-11-27-mega-text-execution-amendment.md)
- [ACTIVE] Fastest remaining job first, supersedes Retrieval-last priority -> [detail](decisions/2026-11-28-fastest-first.md)

## Information Map

- Long-term rules and cumulative knowledge live here; detailed rationale and evidence live in `decisions/`, `sessions/`, and `refs/`.
- Current goal, status, next action, blockers, and validation live in `state.md`.

## Sources

- Link each rule to a relevant decision file when one exists.
