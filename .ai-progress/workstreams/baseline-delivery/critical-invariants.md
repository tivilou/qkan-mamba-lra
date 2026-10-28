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

## Decision Index

<!-- Add one bullet per decision: `- [STATUS] Short title -> [detail](relative/path.md)`. -->
- No decisions indexed yet.

## Information Map

- Long-term rules and cumulative knowledge live here; detailed rationale and evidence live in `decisions/`, `sessions/`, and `refs/`.
- Current goal, status, next action, blockers, and validation live in `state.md`.

## Sources

- Link each rule to a relevant decision file when one exists.
