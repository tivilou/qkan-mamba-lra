# Mamba execution boundaries and finite micro test

- Request: collaborator runs tests; optimize project execution structure along stable-core/plugin boundary. Preserve active fixed executor and seven completed results.
- Baseline:89353fc83f894d5407e358191cbaa4db768ecbc8. Added explicit optional worker hooks, instance-scoped pinned FP32 fusion plugin, science/UI binding separation, latest fused-state migration compatibility. Historical entrypoints retained.
- Added --micro-benchmark to existing mamba-speed diagnose entry: micro8/16/32, logical32, 18finite attempts maximum, isolated exact CUDA OOM, source restoration/owned copy cleanup, compact auto-return. No formal policy changes.
- Validation:41focused stdlib checks passed; actual cached upstream ASTs compiled with fake dependencies, original scientific functions compared to baseline, real supervisor argument boundaries exercised. No owner framework/model/GPU/import or installation.
- Delivery: docs/baselines/mamba-validation.md current command benchmarks latest paused89353fc Suite then resumes its original fixed executor and integrated dashboard. Architecture docs/baselines/execution-architecture.md. Package .runtime/mamba-micro-plugin-20261124 contains candidate/diff/verification/runnable rollback.
- Pending: native throughput/peak memory/adaptive event return and separate v2 native fresh-process recovery. Do not label code delivery as measured acceleration.
