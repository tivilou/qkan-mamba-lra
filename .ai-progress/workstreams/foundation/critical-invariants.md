# Critical Invariants: 研究恢复、协作交付与科研门户

<!--
Keep this file short. Record only current rules that constrain edits,
publication, experiments, data access, workspace boundaries, or claims.
Put rationale and chronology in decisions/*.md instead.
-->

## Scope

- Workstream: `foundation`.

## Mandatory Rules

- Formal training belongs to collaborators; this workstream implements delivery tools, evidence records, progress and portal. Do not launch formal model training.
- Never connect to the released original 3080Ti endpoint. Resolve replacement explicitly with serverctl before any operation; preserve other projects and services.
- Preserve upstream 74acc3d and historical report wording; append corrections rather than rewriting old evidence. Do not synthesize the missing historical JSON, checkpoint or sample traces.
- Keep credentials and endpoint configuration outside Git and portal; all uploads are explicit, namespace-scoped, hash-checked, immutable and recoverable.
- Dataset selection and file existence are not qualification. No formal baseline comparison until data is qualified and the cohort/protocol are frozen.
- Each new formal runner must pass correctness, full-state recovery when long-running, and real sample-trace.v2 structural/semantic/replay gates before handoff.
- Use actual input samples and recorded intermediate values for Case Study; unavailable historical traces remain explicitly unavailable.
- Keep scientific results in source records, and render derived portal data deterministically. No hidden metrics recomputation in the browser.

## Decision Index

<!-- Add one bullet per decision: `- [STATUS] Short title -> [detail](relative/path.md)`. -->
- [CANDIDATE] 协作执行与证据边界 -> [detail](decisions/2026-10-21-collaboration-and-evidence.md)

## Information Map

- Long-term rules and cumulative knowledge live here; detailed rationale and evidence live in `decisions/`, `sessions/`, and `refs/`.
- Current goal, status, next action, blockers, and validation live in `state.md`.

## Sources

- Link each rule to a relevant decision file when one exists.