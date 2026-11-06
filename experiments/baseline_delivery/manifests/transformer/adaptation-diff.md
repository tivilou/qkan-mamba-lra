# Transformer adaptation boundary

Allow project vocabulary/pad mask, scalar pixel projection, positional encoding, masked mean pooling and shared retrieval encoder/pair head. Disclose GELU/prenorm/width/layers/training as project choices differing from original Transformer and some LRA recipes.

No attention approximation, unmasked padding, test tuning or output repair without new review. Measure full-attention resource use at each task length; one bounded retrieval canary does not certify every full run. Stronger Transformer recipes remain a separate landscape consideration.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.

Explicit migration may fork a paused admitted31639c0 full-state checkpoint into a separate adaptive suite after new source/recovery/trace admission and a fresh native-worker zero-update restore/roundtrip check. Only revision/contract and controller metadata change; scientific configuration is locked. The fixed prefix and adaptive continuation retain lineage and must not be reported as an uninterrupted fixed-policy baseline. Original checkpoints are preserved; native migration remains executor validation.
