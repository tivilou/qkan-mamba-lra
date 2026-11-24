# Baseline delivery

## Goal

Deliver faithful, recoverable baselines and measured optional acceleration; preserve current progress, completed results and compact automatic return.

## Current State

- MICRO NATIVE RETURN 2026-11-24: authenticated25202bytes/SHA396bd42afef6b80ed360bd6978e67599573c368153ebbd81756d808fdea4fda0, Suite071532Z-cc3cbe0b, latest source step34149. One clean native child22.48s/18disposable/504fusedcalls/0formal, source unchanged/fullrestore exact/ownedcopy removed. FP32pluginv2 micro8 89.826samples/s;16 77.210 (-14.04%);32 71.985 (-19.86%), neither larger tier faster in any of4pairs. Keepmicro8/logical32. Recent32events contain16physical16to8 throughput reverts, logical32 unchanged. Review docs/evidence/mamba-micro-native-review-20261124.json. No whole-dataset speed/equivalence or v2 formal recovery certification; original post-resume heartbeat not included.

- User screenshot 2026-11-24: fixed executor89353fc83f89, Suite mamba-fused-formal-20261124T061405Z-b57105a1; 7computed/1running/17pending, retrieval step27415/epoch8, micro8/logical32, 80.33samples/s, 4.38/24GiB. Screenshot is an observation, not an authenticated checkpoint or recovery audit. Do not restart from historical24981.
- User requests collaborator-run micro test and structural cleanup. New execution_plugins.py explicitly registers mamba.fused_fp32 v2 and exposes optional worker hooks. New path uses instance-owned forward/private pinned autograd definitions; no shared trainer/upstream global replacement. Original update, gradient, loss, save/load and RNG function ASTs unchanged vs89353fc. Historical diagnostics remain compatible. Architecture guide: docs/baselines/execution-architecture.md.
- v2 scientific binding includes core/model/data/execution dependencies, excludes controller/UI. Old Suites keep original executor/revision. Explicit new v2 preparation supports latest paused fused checkpoint and completed-result preservation, resets only obsolete execution timing windows. Native v2 recovery gate still required; no automatic migration of running experiment.
- Existing diagnose mamba-speed now accepts --micro-benchmark: one finite child, FP32 fused/eager, fixed logical32, micro8/16/32, 6warmups+12balanced timings, latest full-state restored each attempt, exact CUDA OOM isolation, actual fused call count and peak memory. Recommends only >=3% median gain and >=3/4 faster pairs; no formal config change or numerical equivalence claim. Copies only selected checkpoint, removes owned copy, returns bounded adaptive events/report automatically.
- 41 focused standard-library checks passed; no owner torch/numpy/model/GPU execution or install. Evidence: docs/evidence/mamba-micro-plugin-delivery-20261124.json. Runtime package: .runtime/mamba-micro-plugin-20261124 (candidate/change.patch/verification.json/rollback.py). Hardware micro result now received above; native v2 formal save/recovery remains pending.

## Single Next Action

Keep original89353fc fused training at micro8/logical32; do not increase micro or repeat unchanged timing. Native report recommends8 and confirms earlier physical throughput reverts. Existing command resumes original fixed Suite with integrated dashboard after successful test; actual live post-resume heartbeat is not in this report. Future changes should target a demonstrated bottleneck, not memory occupancy alone. v2 formal migration remains separate from this finite benchmark.

## Evidence and Retained Decisions

- FP32 fused native review: docs/evidence/mamba-convolution-native-review-20261124.json, 4clean children/18disposable/0formal; matched update median .413793646s -> .352782322s, +17.2943% throughput, all4pairs faster; tiny nonzero numeric differences, not convergence certification. Fixed pinned offline causal-conv1d1.5.2 wheel and official ABI bridge; preserve shared environment.
- Historical launch/interface/display repairs retained in Git89353fc and scoped sessions. Original Supervisor --output and five-positional-argument Executor bugs are covered by focused tests. Display uses synchronized changed-row writes; screenshots now show successful formal progress.
- BF16 whole-model Mamba not admitted: docs/evidence/mamba-noise-native-review-20261123.json, update differences22.86%-24.49% vs much smaller repeats. Input projection BF16 +0.2816% speed is not useful: docs/evidence/mamba-projection-native-review-20261123.json. No unchanged reruns or automatic BF16/Graph enablement.
- Historical unified entry/display/formal and scientific fidelity decisions remain in critical-invariants.md, decisions/, docs/baselines/experiment.md and Git history. MEGA/S5/S4D execution unchanged; assess separately later.

## Plan Control

plans/optional-optimizations/v001.json: OP01/02/03 done, OP04 skipped, OP05 active, OP06 todo. Finite observations do not establish full convergence. Preserve historical manifests/receipts and their delivery-time claim scopes.
