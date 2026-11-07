# Baseline delivery

## Goal

- Deliver source-pinned recoverable baselines, collaborator-only native execution and automatic artifact return.

## Current State

- Latest user screenshot2026-11-07 11:03 CST: source9a suite adaptive-migrated-20261106T111646Z-90465 has completed Transformer/ListOps and is running Transformer/Text, epoch9/100,162472/1987700 samples, actualstep1475,25.16samples/s, logical128/physical16, cumulativeOOM5, GPU100%,19.84/24GiB. User explicitly has NOT updated to66c8eda. User screenshot only, not independently inspected. Current source checkout from prior delivery is /root/projects/qkan-mamba-lra-adaptive-9a147cd70ced. Old9a code confirms ListOps padded length2048 versus Text4000, dense Transformer without length trimming/nested tensor; attention arithmetic term scales approximately3.81x. Observed slowdown126.45/25.16≈5.03x is plausibly longer sequences plus smaller physical slices, not established by a profiler.
- IMPORTANT HANDOFF CHANGE: do not execute previous migrate_performance.sh instructions now. The published66c8eda migration snapshot explicitly rejects ANY source job with summary.json; this suite now has a completed job. A future migration must preserve completed results with original identities and admit continuation of remaining jobs, without retraining/relabeling the completed one. User has asked only for explanation in this turn; that code change has not been implemented.
- User authorized trying further acceleration. New opt-in --performance adds bounded maximum-logical CPU lookahead with strict sampler-prefix checks to Torch/S5; Torch reuses pinned host storage after existing update synchronization. FP32, logical32/64/128, physical policy, Adam/clip and sample LR policy remain declared. No AMP/TF32/compiler/new attention/dependencies. Timing-dependent future tier choices can differ; no bitwise future equivalence claim.
- First16 committed formal updates yield immutable numeric performance-profile.json. Torch uses CUDA events and host phase timing without extra per-phase synchronization; S5 exposes host/update-wall only. Warmup is included; overlapping phases cannot be summed; failed OOM phase events omitted. Independent bounded uploader sends compact report before suite completion to authenticated performance-profile/<Suite>--<job>; failures retain local retryable files. Final suite return includes report.
- scripts/migrate_performance.sh accepts source9a only, creates separate hash-verified checkpoint snapshot/new suite, and preserves original sample_clock, adaptive controller including OOM ceiling/pending windows, optimizer moments/counters, RNG, permutation/cursor, best/history/time and prior migration lineage. No LR rebasing or warmup restart. Started jobs without full checkpoints and completed source jobs are rejected. Five source probes,25 task recovery/trace gates and zero-update native migration roundtrip precede automatic training supervision. Existing fixed316/adaptive96 migration paths remain separate.
- New handoff: docs/baselines/fp32-performance.md; implementation evidence docs/evidence/fp32-performance-implementation-20261106.json. Historical645 canary failure/prefetch repair evidence retained;9a is now the current source, not96.
- Offline authenticated-exchange-only setup reuses existing environments/cache,68 published wheels and pinned source/header assets. No owner framework import, model execution, GPU query, dependency install or collaborator interruption. No remote experiment checkout changed by publication.
- Committed release remains closed; all five candidate manifests not_verifiable until revision/device/protocol-specific native evidence passes. Main cohort Transformer/Mamba1/MEGA/S5; minimalS4D auxiliary. Dataset internal noncommercial/no redistribution. No benchmark/SOTA/top-two/quantum claim.

## Next Actions

- Continue existing9a run while deciding next acceleration.66c8eda was published via the authorized3080 relay, but its previously supplied migration command is now blocked by the completed ListOps job. Before another handoff, implement preservation of completed original-revision jobs plus migration of unfinished jobs, or an explicitly separate remaining-job suite with linked original results. Never silently retrain or relabel completed results.
- Keep printed new Suite path for future formal_progress.py --resume. Review automatically returned native gates/migration report and first16-update numeric performance report, then compare stable samples/s and final accuracy. Compile/AMP remain future separate decisions after profiling.

## Blockers / Validation

- 86 standard-library/mock checks pass across adaptive execution, logical batch, migration and performance helpers. New checks cover tier-changing prefix prefetch/tails, immutable sampler IDs, pin reuse/fallback, bounded event timing/OOM omission, preserving sample clock/controller, unique nonblocking report upload/startup failure. AST/readiness help and five manifest/provenance/staged-policy checks are required before commit.
- Native CUDA/JAX restore/roundtrip, Linux wrappers, actual memory behavior and measured speedup remain pending collaborator execution. Code delivery does not certify native passage or accuracy.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. Plan one-click/v001.json F01 refers to the earlier admitted revision; formal-result follow-up remains.
