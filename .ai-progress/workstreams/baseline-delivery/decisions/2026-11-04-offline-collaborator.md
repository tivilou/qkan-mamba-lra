# Assume collaborator has no external internet

- Date: 2026-11-04
- Scope: baseline-delivery
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: All required exchange assets published with size/SHA receipts before collaborator handoff; actual GPU gates still required before training
- Authorization: User explicitly authorized full repair and required no external access; no owner model tests, training or dependency installation

## Decision

User requires collaborator scripts to operate with pre-provisioned local data/runtime and authenticated project exchange only. Prepare complete dependency/source assets on the owner side; disable public Conda, pip, Git and author-data fallbacks.

## Reason

Full official URLs are external; the failed Aliyun mapping repair cannot establish reachability. Collaborator screenshots establish local system Python3.10 but missing ensurepip.

## Consequences

Use stdlib venv without pip plus hash-verified local pip wheel; pin complete Linux cp310 wheels and source archives. Missing local prerequisites/assets stop with automatic private redacted diagnostics. Preserve runtime fidelity/admission gates.
