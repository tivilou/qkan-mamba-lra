# Minimal S4D adaptation boundary

Allow hash-checked DropoutNd import substitution, task embeddings/projection, project prenorm/residual/dropout stack, padding mask, masked pooling and classification/pair head. Preserve the official mixer and optimizer metadata.

Report project training separately from original S4 recipes. Never rename minimal S4D to full S4/S5 or substitute historical src/baselines/s4.py. No output repair or test-based selection. Original scores are not directly comparable to document-disjoint-v3.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.
