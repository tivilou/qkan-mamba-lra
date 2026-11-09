# Baseline delivery

## Goal

- Deliver source-pinned recoverable baselines and beneficial native compiler/Graph execution, preserving training progress, original completions and automatic return.

## Current State

- Current reviewed runtime uses compiler policy v5: compatible Adam rollback storage reuse, synchronized/counterbalanced complete-update measurement and bounded no-gain phase diagnostics. Native v5 benefit and sustained speed remain unverified. Release stays closed; five manifests remain not_verifiable. Publication uses only the authorized3080 Git/network relay.
- Latest authenticated native return: formal-readiness/failure-37b2622a59ef, failure.json SHA256bf23d85bef0362f6777e75c12be6069b5e3de0ef7128b15b93089f5e662a1f72. Source d808f0a4dbc843301fc4849d8cbaa27de5ecc6e1, preparation Suite runs/adaptive-migrated-20261109T134728Z-225944. Shell returned at /root/projects/qkan-mamba-lra-adaptive-d808f0a4dbc8; formal continuation did not start.
- d808 Graph/Dynamo native autograd passed synthetic exact recovery with400/448replays and REAL checkpoint full-update numerical/RNG comparison. Real Graph replay count80, stage ready, no capture diagnostics. Rejection is benchmark_no_gain: eager8.7803204656s versus Graph8.8010682687s/update, approximately0.236%slower over two short pairs. Exact rollback,0samples/updates committed. This establishes real Graph correctness, not beneficial or sustained acceleration. Inductor still fails exact recovery; AOT still fails numerical/RNG checks. Compact evidence: docs/evidence/compiler-calibration-measurement-20261109.json.
- v5 rollback preflights the entire optimizer state, packed parameter order and tensor shape/type, then copies snapshot values into existing live storage and separately restores real param-group controls; no snapshot aliasing. Lazy/new state retains detached-load fallback; native OOM retains CPU rollback. Preparation allocation/cache overhead is reduced, but formal per-update speed improvement is not established.
- Timing now synchronizes queued restore work before starting each timer, measures two finite AB/BA pairs after one warmup, records raw durations/order/gain and still requires positive median complete-update benefit. No retry-until-lucky. If correct but no gain, profile exactly one additional same-snapshot update per mode, excluded from admission measurement, then exact zero-commit rollback. Automatic return contains forward/loss, backward and clip/optimizer/schedule host/stream times plus allocator summaries; terminal explains measured rejection. These observations must guide the next actual kernel optimization.
- Retain v4 two-document Graph capture for Transformer/Retrieval: torch.compile backend=eager/fullgraph with native autograd, separate document graph pools, native pair head, shared parameters and unchanged FP32/dropout/equations. This is FX/Graph execution, not Inductor fusion. Per-document uint32 Philox bound fits micro16 but rejects micro32 before capture; capacity downgrade affects physical partition only, persists a no-reprobe ceiling and stays separate from genuine OOM. Previous source-grounded capacity inference and historical evidence remain in docs/evidence/compiler-graph-rng-capacity-20261109.json.
- Required Retrieval Graph still needs unchanged exact fresh-process recovery/trace, real full-update rtol1e-5/atol1e-6 with exact RNG/controls, exact zero-commit rollback, actual replay and positive measured benefit. No silent eager long run. Shape/generation checks,8build limit, native tails and finite OOM retries retained. S5 remains native JAX.
- Actual training migration source remains revision840430a414bcf8bd6eb7c4b9c105e689578b1780, Suite /root/projects/qkan-mamba-lra-adaptive-840430a414bc/runs/adaptive-migrated-20261109T045101Z-183941, cache /root/qkan-cache,GPU0. Latest authentic snapshot restored Retrieval step12683, checkpoint SHA25695b8a45e0091d1e93596fc2ce00c4a925aa9b7a47c7a3664b0639e254a51a7f6. Never migrate from failed d808/e07/912 preparation Suites or historical9ce/9a. Original840 was safely paused; do not remotely interrupt collaborators.
- Preserve transformer-listops-s42 and transformer-text-s42 result bytes and original9a147cd70cedf1511de89d0381dfbdabac3274f2 provenance. Migration preserves model/Adam/RNG/sampler/cursor/sample-clock/controller/lineage; only old compiler selection is archived/reset.25jobs=5models*5tasks seed42,100epochs;20main+5minimalS4Dauxiliary. No sample-budget change or fixed-batch equivalence claim.
- Dashboard retains60s rolling committed-sample rate and process-local long average, excluding restored prefix; stale>120s marks old rates and pauses ETA. ETA is current task only. Display never benchmarks or restarts training.
- Runtime/install remains offline/local/authenticated-exchange-only, with bounded automatic diagnostics/receipts. No owner framework/model imports, GPU queries, dependency installation or training, locally or3080. Data/weights/raw logs/traces/secrets stay outside public Git. No SOTA/top-two/quantum claim.

## Next Actions

- Deliver docs/baselines/torch-acceleration.md command: start at current d808 checkout, fetch a new isolated executor and migrate original840 full checkpoint with --require-graph transformer-retrieval-s42. All remaining gates and real calibration must pass before formal continuation; two original completions retained. Do not contact junior directly.
- Read automatic return. If no gain persists, compare separate forward/backward/optimizer phase evidence; do not mislabel a capture fix or timer change as acceleration. Same-version normal resume uses the admitted supervisor, without repeated migration/readiness.

## Validation / Evidence

-215 stdlib/mock checks passed, including11 new timing/storage/diagnostic tests: rollback copy completion excluded from timers, AB/BA order, no retry-until-gain, diagnostics excluded and fully rolled back, live Adam device/storage/parameter identity, snapshot independence and preflight fallback. No native framework/device execution.
- Pinned official PyTorch v2.7.1 optimizer source inspected via3080: packed state values reference original state; packed group metadata is copied. Source hash, authentic d808 finding and implementation hashes recorded in docs/evidence/compiler-calibration-measurement-20261109.json; prior evidence preserved.
- Five source manifests/contracts/provenance and data/release/portal boundaries pass; readiness/display CLI, Bash syntax, changed Python AST and diff whitespace pass. Exact staged policy/artifact identities checked before commit.

## Relevant Decisions

- decisions/2026-11-04-offline-collaborator.md; decisions/2026-11-04-implementation-roster.md. one-click/v001.json F01 records an earlier admitted launch; current Graph follow-up remains open.
