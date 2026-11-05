# Adaptive memory execution

This opt-in execution policy adjusts the physical microbatch while holding the logical batch (32 by default), dataset/order, learning-rate schedule, optimizer-update count, clipping and FP32 precision fixed. It covers the PyTorch Transformer/Mamba1/MEGA/minimal-S4D worker and the native JAX S5 worker. It introduces no dependency downloads or framework replacement. Each job owns an independent controller.

Changing microbatch partitioning can change dropout keys, reduction order and numerical trajectories. Adaptive configurations are distinct experiments, not bitwise-equivalent continuations of the existing fixed-microbatch suite. Current executor31639c0 and its active suite must continue untouched. Do not git pull there or alter its checkpoint/config to enable adaptation.

## Policy

- Default tiers are logical-batch divisors: 1/2/4/8/16/32 for batch32. Start at the previously validated physical batch4, rather than treating untested32 as known safe. Batch size never exceeds the logical batch.
- Discard two warmup timings and compare eight complete logical updates per window. Low memory occupancy enables a trial of the next tier only when estimated peak fits within85% of device capacity and free memory covers the predicted increment plus at least512MiB. Missing telemetry disables promotion, not training.
- Keep a larger tier only if its measured complete-update throughput improves at least5%. Otherwise return to the previous tier and wait256 measured updates before another trial. Accepted tiers may continue climbing. A current trial's updates are real training updates, not extra optimization steps or validation/test tuning.
- On a recognized allocation OOM, drop one tier, reset the pending throughput trial and apply the cooldown. Restrict the ceiling until the cooldown expires, allowing later promotion if pressure has eased. At most8 retries per logical update or evaluation pass, bounded further by the finite descending tier list. OOM at tier1 fails clearly. Unrelated exceptions never trigger memory retries.
- The controller is a best-effort throughput search, not a guarantee of global optimal performance. Window noise, changing sequence cost, CPU copies, JIT recompilation and external GPU users can affect it. Whole-run training_seconds includes adaptation overhead; measured update comparisons include snapshot/retry work but discard warmup updates. Benchmark against the fixed-policy baseline before claiming speedup.

## OOM transaction and resume

PyTorch captures model parameters/buffers, optimizer, scheduler and Python/NumPy/Torch/CUDA RNG to a bounded CPU rollback snapshot before each logical update. Failed tensors/gradients are released, only this process's unused allocator cache is cleared, and all captured state is restored before retrying the same sample IDs. The snapshot cost is included in the update timing. The cursor, metrics and optimizer step are committed only after the full update and CUDA synchronization succeed. Even an OOM after a partial optimizer update is rolled back.

JAX uses immutable, non-donated TrainState/optimizer arrays. An update computes a candidate state and local RNG; neither replaces the committed state/RNG until device execution succeeds. On OOM it drops failed references, clears local compilation caches and retries from the unchanged state. Compilation can recur at new shapes. No unrelated process is killed or cache cleared.

Evaluation OOM discards the incomplete evaluation pass and retries with a smaller physical batch. Accuracy-based model selection and the selected-checkpoint validation reproduction check remain mandatory. Model/static-state initialization, checkpoint storage errors and trace capture are not solved by training microbatch tiering; those failures propagate. In-process rollback does not recover host OOM, fatal CUDA errors or process termination. Those use the existing atomic full-state checkpoint; work after the last durable generation can repeat.

Controller state (tier, failed-tier ceiling, samples, warmup, cooldown, pending trial and cumulative OOM count) is stored inside each full-state checkpoint. The advisory adaptive-memory.json never replaces checkpoint-owned state on restore. Hardware pressure and wall-clock observations are external inputs: an uninterrupted run and a resumed run are not promised identical future adaptive decisions under different observations.

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

## Evidence and display

Per-job heartbeat and the CLI show the physical batch used for the completed update, next tier, cumulative OOM retries and the latest transition. adaptive-memory.json contains current policy/controller and numeric memory telemetry. adaptive-events.jsonl appends transitions with source/destination tiers, step, timestamp and numeric memory readings. Summary records the policy and adaptive-trajectory label. Compact diagnostics accompany automatic failure return; control state/events accompany completed formal return. No inputs, sample text, labels, raw exceptions or tensors are added to these diagnostics.

GPU memory and utilization readings are observability inputs, not instructions to fill VRAM. There is one worker per physical GPU; adaptation does not silently introduce concurrency, AMP, TF32 or a larger logical batch.
