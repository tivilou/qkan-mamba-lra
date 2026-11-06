# Adaptive memory execution

This opt-in execution policy adjusts the physical microbatch while holding the logical batch (32 by default), dataset/order, learning-rate schedule, optimizer-update count, clipping and FP32 precision fixed. It covers the PyTorch Transformer/Mamba1/MEGA/minimal-S4D worker and the native JAX S5 worker. It introduces no dependency downloads or framework replacement. Each job owns an independent controller.

Changing microbatch partitioning can change dropout keys, reduction order and numerical trajectories. Adaptive configurations are distinct experiments, not bitwise-equivalent continuations of the existing fixed-microbatch suite. Keep executor31639c0 and its checkpoint/config untouched. The explicit migration entry below can fork its paused full state into a separate adaptive suite; it never edits the original contract in place.

## Policy

- Default tiers are logical-batch divisors: 1/2/4/8/16/32 for batch32. Start at the previously validated physical batch4, rather than treating untested32 as known safe. Batch size never exceeds the logical batch.
- Discard two warmup timings and compare sixteen complete logical updates per window. Low memory occupancy enables a trial of the next tier only when estimated peak fits within85% of device capacity and free memory covers the predicted increment plus at least512MiB. Missing telemetry disables promotion, not training.
- There is no minimum percentage gain. Keep a confirmed positive improvement, including1%: at least three faster windows, at least two-thirds of observed windows faster, and a faster overall mean. A single slow window never triggers reversion. Revert only after at least three slower windows, a two-thirds slower majority and a slower overall mean; then wait256 measured updates before another trial. These are noise guards, not a statistical significance test.
- Extend mixed/equal observations up to eight windows; retain an inconclusive tier with cooldown, without claiming a speed gain. Accepted tiers may continue climbing. A current trial's updates are real training updates, not extra optimization steps or validation/test tuning.
- On a recognized allocation OOM, drop one tier, reset the pending throughput trial and apply the cooldown. Restrict the ceiling until the cooldown expires, allowing later promotion if pressure has eased. At most8 retries per logical update or evaluation pass, bounded further by the finite descending tier list. OOM at tier1 fails clearly. Unrelated exceptions never trigger memory retries.
- The controller is a best-effort throughput search, not a guarantee of global optimal performance. Window noise, changing sequence cost, JIT recompilation and external GPU users can affect it. Whole-run training_seconds includes adaptation overhead; measured update comparisons include batch preparation/transfer, snapshot/retry work and synchronization but discard warmup updates. Advisory-file writes happen after the measured update; their time remains included in training_seconds. Benchmark against the fixed-policy baseline before claiming speedup.

## Execution overhead

The adaptive policy prepares one complete logical batch on the CPU, validates token padding there and stages it on the GPU once. Each microbatch slices that staged batch. A single CPU thread prefetches at most one subsequent batch using immutable sampler IDs; it performs no device work and shuts down on normal exit, pause or failure. Prefetch neither advances the committed sampler nor changes sample order. PyTorch uses pinned-memory/nonblocking transfers when available. Whole-batch allocation OOM falls back to CPU staging with microbatch transfers.

Loss/accuracy accumulate on the device and become host scalars once per logical update. Finite-loss/gradient checks remain before optimizer mutation. S5 checks gradient finiteness on device instead of copying every gradient to the host. Advisory heartbeat and adaptive-memory.json writes occur at most once every five seconds during steady training, with immediate writes on tier transitions or safe pause. Full-state checkpoint cadence is unchanged.

These optimizations apply to the opt-in adaptive execution path. FP32, the logical batch, clipping, update count and learning-rate schedule stay declared in the configuration. AMP, TF32 and graph compilation are not enabled by this policy. Actual throughput and device-allocation behavior await executor checks.

## OOM transaction and resume

PyTorch captures model parameters/buffers, optimizer, scheduler and Python/NumPy/Torch/CUDA RNG before each logical update. Reusable same-device tensor buffers avoid copying the full GPU state to the CPU every update; CPU RNG values stay on the CPU. Backup storage contributes to measured GPU memory pressure. If backup allocation or restoration exhausts GPU memory, the job switches to CPU backups. Rare optimizer restoration uses an independent host copy to prevent the live moments from aliasing reusable backup tensors. Failed tensors/gradients are released, only this process's unused allocator cache is cleared, and all captured state is restored before retrying the same sample IDs. The snapshot cost is included in the update timing. The cursor, metrics and optimizer step are committed only after the full update and CUDA synchronization succeed. Even an OOM after a partial optimizer update is rolled back.

JAX uses immutable, non-donated TrainState/optimizer arrays. An update computes a candidate state and local RNG; neither replaces the committed state/RNG until device execution succeeds. On OOM it drops failed references, clears local compilation caches and retries from the unchanged state. Compilation can recur at new shapes. No unrelated process is killed or cache cleared.

Evaluation OOM discards the incomplete evaluation pass and retries with a smaller physical batch. Accuracy-based model selection and the selected-checkpoint validation reproduction check remain mandatory. Model/static-state initialization, checkpoint storage errors and trace capture are not solved by training microbatch tiering; those failures propagate. In-process rollback does not recover host OOM, fatal CUDA errors or process termination. Those use the existing atomic full-state checkpoint; work after the last durable generation can repeat.

Controller state (tier, failed-tier ceiling, samples, confirmation windows, warmup, cooldown, pending trial and cumulative OOM count) is stored inside each full-state checkpoint. The CPU-backup fallback choice is checkpoint-owned too. The advisory adaptive-memory.json never replaces checkpoint-owned state on restore. Prefetched data are reconstructed from the saved cursor. Hardware pressure and wall-clock observations are external inputs: an uninterrupted run and a resumed run are not promised identical future adaptive decisions under different observations.

## Launch boundary

Use a clean separate checkout of this revision with the existing local/exchange prerequisites available; preserve the active executor checkout. Do not enable this policy by editing an old suite. Existing setup and pinned sources remain offline/exchange-only.

An optional code-only helper can be fetched into the active checkout's ignored .bootstrap directory. It fetches the branch, creates a sibling detached worktree named by its revision and links the existing local environments/bootstrap assets. It does not launch a GPU job or modify active tracked files. Existing mismatched targets/assets are rejected. Shared pinned assets are reused, not rebuilt for this policy.

```bash
cd /root/projects/qkan-mamba-lra
git fetch origin research/collaboration-foundation &&
git show FETCH_HEAD:scripts/prepare_adaptive_checkout.sh > .bootstrap/prepare-adaptive-checkout.sh &&
bash .bootstrap/prepare-adaptive-checkout.sh
```

Use the new path printed by the helper. Schedule admission on an available GPU; do not overlap a probe with the current formal worker on GPU0 just to upgrade code.

```bash
# In the NEW checkout, first run admission without starting formal training:
bash scripts/run_adaptive_readiness.sh --cache "$QKAN_CACHE" --gpus 0 --gates-only
```

The command prints the actual new Suite path. It runs the source probes and all25 bounded recovery/trace canaries, binding adaptive configuration, source, device and controller restoration. A canary covers its observed schedule; it is not an exhaustive test of every pressure/timing trajectory. Owner-side injected-failure tests cover controller/transaction logic without importing models. Actual Torch/JAX rollback, speed and all-tier numerical behavior remain hardware checks until executed and reviewed.

After successful admission, use that printed suite path with the same wrapper and --resume to continue the new experiment. The formal_progress supervisor preserves --adaptive-memory from its context on subsequent resumes. Certificates from31639c0 cannot authorize this revision; the original release remains closed in Git.

## Fork an existing fixed checkpoint

The explicit migration entry supports the admitted31639c0 suite before any job has completed. Pause its original supervisor with Ctrl+C once and wait for the shell prompt. Source suite/worker locks must be free. The importer rejects completed jobs rather than relabeling old fixed results as adaptive. Started jobs without valid full checkpoints stop migration; never silently restart them. Untouched pending jobs start normally in the new adaptive suite.

Run in the separate checkout created by prepare_adaptive_checkout.sh:

```bash
bash scripts/migrate_fixed_to_adaptive.sh \
  --migrate-from /root/projects/qkan-mamba-lra/runs/formal-ready-20261105T042214Z-1e90315e \
  --cache /root/qkan-cache --gpus 0
```

This snapshots source configuration and the latest generation bytes with SHA-256 validation. It reruns this revision's source/recovery/trace gates and obtains new admission. Then a fresh native worker loads each copied full checkpoint, verifies the original data/dictionary contract and unchanged scientific configuration, restores parameters/optimizer/scheduler/RNG, checks exact restoration/serialization roundtrip and writes a new checkpoint. This check performs zero optimizer updates. No old certificate is copied into the new release. The wrapper enters the in-place progress supervisor only after all checks and evidence return succeed, then continues the remaining update budget.

Migration preserves committed step/epoch/cursor/permutation, all optimizer moments/counters, scheduler, RNG, best weights/validation history and accumulated training time. Only source/config contract metadata is rebound and a new adaptive controller is initialized. Scalar runtime/package/backend identity must still match. Changing model dimensions, data, logical batch, epochs, seed, optimizer settings or protocol is rejected. The new run is a fixed-prefix/adaptive-continuation trajectory with checkpoint hash and switch step in migration-validation.json, gate-evidence.json and final summary. It is not an uninterrupted fixed baseline.

If a gate fails, do not rerun the migration wrapper to duplicate an experiment: keep the printed new Suite path, inspect its local logs, and resume its admission with run_adaptive_readiness.sh --resume NEW_SUITE --cache /root/qkan-cache --gpus 0 --gates-only after resolving the cause. After a successful admission use formal_progress.py --resume NEW_SUITE. A partially written target checkpoint without migration-validation.json is blocked for explicit inspection. Original checkpoints remain available for rollback through the original executor. Do not run both suites on GPU0 simultaneously.

Owner verification covers stdlib migration policy, byte-copy/hash behavior and injected failures only. Native restore checks and new GPU admission execute on the collaborator; implementation availability does not establish that this particular checkpoint has already migrated successfully.

## Evidence and display

Per-job heartbeat and the CLI show the physical batch used for the completed update, next tier, cumulative OOM retries and the latest transition. adaptive-memory.json contains current policy/controller and numeric memory telemetry. adaptive-events.jsonl appends transitions with source/destination tiers, step, timestamp and numeric memory readings. Summary records the policy and adaptive-trajectory label. Compact diagnostics accompany automatic failure return; control state/events accompany completed formal return. No inputs, sample text, labels, raw exceptions or tensors are added to these diagnostics.

GPU memory and utilization readings are observability inputs, not instructions to fill VRAM. There is one worker per physical GPU; adaptation does not silently introduce concurrency, AMP, TF32 or a larger logical batch.
