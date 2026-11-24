# Baseline delivery

## Goal

Deliver faithful, recoverable baselines and measured optional acceleration; preserve current progress, completed results and compact automatic return.

## Current State

- User screenshot 2026-11-24: fixed executor89353fc83f89, Suite mamba-fused-formal-20261124T061405Z-b57105a1; 7computed/1running/17pending, retrieval step27415/epoch8, micro8/logical32, 80.33samples/s, 4.38/24GiB. Screenshot is an observation, not an authenticated checkpoint or recovery audit. Do not restart from historical24981.
- User requests collaborator-run micro test and structural cleanup. New execution_plugins.py explicitly registers mamba.fused_fp32 v2 and exposes optional worker hooks. New path uses instance-owned forward/private pinned autograd definitions; no shared trainer/upstream global replacement. Original update, gradient, loss, save/load and RNG function ASTs unchanged vs89353fc. Historical diagnostics remain compatible. Architecture guide: docs/baselines/execution-architecture.md.
- v2 scientific binding includes core/model/data/execution dependencies, excludes controller/UI. Old Suites keep original executor/revision. Explicit new v2 preparation supports latest paused fused checkpoint and completed-result preservation, resets only obsolete execution timing windows. Native v2 recovery gate still required; no automatic migration of running experiment.
- Existing diagnose mamba-speed now accepts --micro-benchmark: one finite child, FP32 fused/eager, fixed logical32, micro8/16/32, 6warmups+12balanced timings, latest full-state restored each attempt, exact CUDA OOM isolation, actual fused call count and peak memory. Recommends only >=3% median gain and >=3/4 faster pairs; no formal config change or numerical equivalence claim. Copies only selected checkpoint, removes owned copy, returns bounded adaptive events/report automatically.
- 41 focused standard-library checks passed; no owner torch/numpy/model/GPU execution or install. Evidence: docs/evidence/mamba-micro-plugin-delivery-20261124.json. Runtime package: .runtime/mamba-micro-plugin-20261124 (candidate/change.patch/verification.json/rollback.py). Actual hardware micro result and native v2 save/recovery remain pending collaborator.

## Single Next Action

Collaborator safely pauses current89353fc runner once and waits for exit; execute current command in docs/baselines/mamba-validation.md. Prepare isolated latest checkout, benchmark latest fused Suite, auto-return report, then resume original89353fc Suite with original same-terminal dashboard on successful measurement. Failure leaves source paused; guide includes original resume command. Do not start another formal Suite or repeat BF16/noise/596check suite.

## Evidence and Retained Decisions

- FP32 fused native review: docs/evidence/mamba-convolution-native-review-20261124.json, 4clean children/18disposable/0formal; matched update median .413793646s -> .352782322s, +17.2943% throughput, all4pairs faster; tiny nonzero numeric differences, not convergence certification. Fixed pinned offline causal-conv1d1.5.2 wheel and official ABI bridge; preserve shared environment.
- Historical launch/interface/display repairs retained in Git89353fc and scoped sessions. Original Supervisor --output and five-positional-argument Executor bugs are covered by focused tests. Display uses synchronized changed-row writes; screenshots now show successful formal progress.
- BF16 whole-model Mamba not admitted: docs/evidence/mamba-noise-native-review-20261123.json, update differences22.86%-24.49% vs much smaller repeats. Input projection BF16 +0.2816% speed is not useful: docs/evidence/mamba-projection-native-review-20261123.json. No unchanged reruns or automatic BF16/Graph enablement.
- Historical unified entry/display/formal and scientific fidelity decisions remain in critical-invariants.md, decisions/, docs/baselines/experiment.md and Git history. MEGA/S5/S4D execution unchanged; assess separately later.

## Plan Control

plans/optional-optimizations/v001.json: OP01/02/03 done, OP04 skipped, OP05 active, OP06 todo. Finite observations do not establish full convergence. Preserve historical manifests/receipts and their delivery-time claim scopes.
