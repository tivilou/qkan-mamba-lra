# Baseline delivery

## Goal

- Deliver source-pinned recoverable baselines and beneficial native compiler/Graph execution, preserving current progress, original completions and automatic return.

## Current State

- Policy v4 repair implemented and verified with stdlib/mocks; native correctness, Graph admission and sustained speed remain pending. Publish this delivery through the authorized3080 Git relay and verify exact GitHub HEAD before handoff. Release stays closed; five manifests remain not_verifiable.
- Latest authenticated native return: formal-readiness/failure-580268ade788, failure.json SHA256163ab49607a74a94b230e69e70059383cee174218b9ffdfd7c0d60cb60c84912. Revisione07a91c7876edd20ae7c320be12339cb0c409c21, preparation Suite runs/adaptive-migrated-20261109T122440Z-221992. Screenshot shell returned at /root/projects/qkan-mamba-lra-adaptive-e07a91c7876e; formal continuation did not start.
- e07 Graph/Dynamo native autograd passed synthetic exact recovery with200/224replays. Real checkpoint micro16 failed at graph_capture, RNG category, torch/nn/functional.py6410 SDPA, before first replay/comparison/benchmark;0updates/samples committed and exact rollback. Inductor still differs in exact recovery; AOT still disagrees numerically/RNG. Reviewed details: docs/evidence/compiler-graph-rng-capacity-20261109.json; earlier repair evidence preserved.
- Source-grounded leading cause, not literal-exception confirmation: pinned PyTorch v2.7.1 limits captured Philox intra-graph offsets to uint32; efficient/cuDNN SDPA reserves B*heads*Lq*Lk per layer. Whole Retrieval micro16 with two4000-token docs and4layers/4heads needs8,192,000,000 for attention alone, beyond4,294,967,295. Synthetic micro4 fits. Old return retained no raw error text; do not claim original offset-overflow message confirmed or OOM observed.
- v4 Graph/Dynamo candidate compiles shared encode with backend=eager/fullgraph and captures document1/2 separately in independent pools, native pair features/head outside. Parameter keys, operation order, dropout, FP32 and model equations retained. Other profiles retain whole-forward capture; S5 remains nativeJAX. FX is not Inductor fusion.
- Before capture, conservative native attention/FF/residual RNG budget is checked for both document shapes. At micro16 each doc bound4,292,608,000 fits; micro32 does not. Explicit pre-capture capacity events use full-update rollback and smaller physical batch, preserve same logical samples/next logical batch, persist a capacity ceiling across cooldown/restart, and are counted separately from OOM. Arbitrary capture/RNG errors are not reclassified as capacity/OOM. Fresh-worker recovery also compares the new optional capacity decisions.
- Required Retrieval Graph still needs exact fresh-process recovery/trace, real full-update rtol1e-5/atol1e-6 with exact RNG/controls, zero-commit exact rollback, actual replay and measured positive complete-update gain. No standards relaxed and no silent eager long run. Shape/generation checks,8build limit, native rare tails, finite native OOM retries and stage/type/category diagnostics retained.
- Actual training migration source remains revision840430a414bcf8bd6eb7c4b9c105e689578b1780, Suite /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941, cache /root/qkan-cache,GPU0. Latest authentic snapshot restored Retrieval step12683, checkpoint SHA25695b8a45e0091d1e93596fc2ce00c4a925aa9b7a47c7a3664b0639e254a51a7f6. Never migrate from failed e07/912 preparation Suites or historical9ce/9a and lose840 progress. Original840 was safely paused before migration; do not remotely interrupt collaborators.
- Preserve transformer-listops-s42 and transformer-text-s42 result bytes and original9a147cd70cedf1511de89d0381dfbdabac3274f2 provenance. Migration preserves model/Adam/RNG/sampler/cursor/sample-clock/controller/lineage; only previous compiler selection is archived/reset.25jobs=5models*5tasks seed42,100epochs;20main+5minimalS4Dauxiliary. No cohort/sample-budget changes or fixed-batch equivalence claim.
- Dashboard retains60s rolling committed-sample rate and process-local long average, excludes restored checkpoint prefix, marks stale>120s, pauses stale ETA and never triggers benchmarking. ETA is current task only. New Graph capacity downgrade count is distinct from OOM retry count.
- Runtime/install remains offline/local/authenticated-exchange-only, with automatic bounded diagnostics/receipts. No owner model/framework imports, GPU queries, dependency installation or training, locally or3080.3080 is an authorized network/Git relay only. Data/weights/raw logs/traces/secrets stay outside public Git. No SOTA/top-two/quantum claim.

## Next Actions

- Deliver docs/baselines/torch-acceleration.md command: start at current e07 checkout, fetch new isolated executor and migrate original840 latest full checkpoint with --require-graph transformer-retrieval-s42. Only passing all remaining recovery/trace/real-calibration gates starts formal training; two original completions are retained. Do not contact junior directly.
- Read automatic return after collaborator execution. Distinguish synthetic success from real-checkpoint correctness, actual replay, positive complete-update gain and sustained short/long throughput. Same-version ordinary resume uses the admitted supervisor; do not remigrate/reenter readiness unnecessarily.

## Validation / Evidence

-204 stdlib/mock checks passed, including16 new partition/RNG-budget/rollback/capacity-ceiling tests. Existing migration/completion preservation, exact rollback, real readiness route and compiler admission checks remain. No native framework/device execution.
- Pinned official v2.7.1 generator, Graph lifecycle, attention and dropout sources inspected through3080 network relay. Compact source hashes, authentic return, implementation hashes and native-pending boundary: docs/evidence/compiler-graph-rng-capacity-20261109.json. Historical evidence retained.
- Five source manifests/contracts/provenance and data/release/portal boundaries pass; readiness/display CLI, Bash syntax, AST and diff whitespace pass. Exact staged policy/artifact identities are checked before commit.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. one-click/v001.json F01 records an earlier admitted launch; current Graph follow-up remains open.
