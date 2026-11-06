# Baseline delivery

## Goal

- Deliver source-pinned recoverable baselines, collaborator-only native execution and automatic artifact return.

## Current State

- Latest user screenshot: suite `adaptive-migrated-20261106T042749Z-67281`, executor96e542dd527b777d7690f6602fa1d14530957f85, Transformer/ListOps Epoch41/100, step121584/300000 (40.53%), physical32/logical32, about3.152 step/s, GPU100%,11.70/24GiB, running1/pending24/failed0. This supports advancing migrated training as reported by the user; it is not independent live inspection, controlled speed evidence or a completed result. Source checkout from prior handoff is `/root/projects/qkan-mamba-lra-adaptive-96e542dd527b`.
- User authorized dynamic logical batch implementation. New opt-in logical32/64/128 plus physical1/2/4/8/16/32/64/128 controller compares seconds/sample, accepts any confirmed positive throughput gain, retains one-window slow trials and reverts confirmed sustained slowdown. Headroom is bounded to85%/512MiB; not a VRAM-filling/global-optimum guarantee. OOM retries identical rows with smaller physical slices; only after commit does the next logical batch decrease. Tier1/retry exhaustion fails clearly.
- Sample-clock LR/progress preserves epochs*train_rows and committed cursor, anchors migration at existing group LR and sample position without warmup restart or counter rescaling. Actual Adam state/counter, RNG, permutation/cursor, history/best weights and accumulated time persist. Logical and physical policies, pending comparison windows and sample clock are checkpoint-owned. Changing batch/schedule defines a distinct optimization trajectory; no fixed-batch equivalence or accuracy claim.
- `scripts/migrate_dynamic_logical.sh` accepts paused admitted fixed31639c0 or adaptive96e542d source suites and creates a separate hash-verified snapshot/new suite. It preserves prior migration lineage, performs new five-source/25-task recovery+trace admission, zero-update native restore/roundtrip, automatic evidence return and supervision. Dynamic canaries force logical32/64/128 twice, interrupt after update2 and exercise128 after fresh-process restore. Formal jobs cannot force the canary sequence. Native validation of this revision is pending.
- Dashboard retains in-place refresh and GPU/microbatch telemetry; dynamic runs add current/next logical batch, sample progress, samples/s, actual Adam step and current-task remaining sample ETA. Old300000-step denominator is not used for the new strategy. Existing physical-only policy remains separately available. See docs/baselines/dynamic-logical-batch.md and docs/evidence/dynamic-logical-batch-implementation-20261106.json.
- Offline/exchange-only setup retains68 published wheels, pinned MEGA/S5 archives, headers and verified receipts. No public-source fallback, new dependency installation or replacement framework. Existing local environments and qualified task cache are reused. No owner framework/model import, GPU query, training or live interruption.
- Existing31639c0 native admission evidence remains historical and valid only for its recorded source/device/protocol (docs/evidence/formal-readiness-passed-20261105.json). Committed release remains closed and all five candidate manifests not_verifiable. Native MEGA/S5 are source-faithful task adapters; main cohort Transformer/Mamba1/MEGA/S5, minimalS4D auxiliary. Internal noncommercial/no-data-redistribution dataset scope; no author benchmark, SOTA/top-two or quantum claim.
- GitHub publication uses the user-authorized development server solely as an SSH SOCKS relay, with existing credentials retained locally, strict host verification and exact remote branch-SHA verification. No remote experiment checkout is changed by publishing.

## Next Actions

- User pauses current supervisor with Ctrl+C once and waits for shell prompt. Fetch code-only prepare helper into ignored .bootstrap, create a new revision-pinned sibling checkout and execute migrate_dynamic_logical.sh against `/root/projects/qkan-mamba-lra-adaptive-96e542dd527b/runs/adaptive-migrated-20261106T042749Z-67281`, cache `/root/qkan-cache`, GPU0. Exact chained commands are in dynamic-logical-batch.md. Do not pull source tracked files or overlap old/new GPU jobs. Keep the printed new suite for later formal_progress resumes.
- Review automatically returned new gate/migration evidence, then actual throughput/accuracy/completed results. Started jobs lacking full checkpoints and already completed source jobs are rejected; no silent fresh fallback or result relabeling.

## Blockers / Validation

- No active-run failure reported. Dynamic native Torch/JAX restore/OOM/allocation behavior, Linux wrappers and measured speed/accuracy await collaborator execution; code availability does not certify them.
- 74 standard-library/mock checks pass:42 adaptive/execution/migration regressions plus32 joint-batch, OOM/sample-clock rollback, migration lineage, forced-tier isolation and dashboard checks. All baseline AST and readiness help pass. Manifests/policy hashes are refreshed and checked separately against staged bytes; no framework execution is used for these checks.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 refers to the earlier admitted revision; ongoing formal-result follow-up remains.
