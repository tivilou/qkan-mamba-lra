# Agent Instructions

<!-- project-progress:start -->
## Project Progress

Use the `project-progress` protocol for durable cross-agent memory.

At the start of meaningful work:

1. Route the request from `.ai-progress/manifest.json`.
2. Read `.ai-progress/project.md`, the selected workstream's `critical-invariants.md`, and its `state.md`.
3. Read decisions only when relevant. Critical invariants are mandatory session-start context. Do not read sessions, refs, archives, or all workstreams by default.

Before finishing, update progress only when durable state changed:

1. Rewrite the selected `state.md` as a concise current snapshot.
2. Record one session outcome for meaningful changes, investigations, decisions, or topic lifecycle transitions.
3. Add decisions only when they should constrain future work.
4. Capture a git snapshot and commit only the agent-owned file changes without adding agent authorship or co-author trailers.
5. Never store secrets, raw logs, full transcripts, or private chain-of-thought.

Treat code, tests, and artifacts as the source of truth. Re-read state before editing and preserve concurrent changes.
<!-- project-progress:end -->

