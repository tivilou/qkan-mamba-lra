# Mamba1 adaptation boundary

Keep the pinned upstream mixer equations/initialization. Declare task embeddings (pad=1), scalar pixel projection, project LayerNorm/residual/dropout stack, post-block padding mask, final masked mean pooling and classification head. Retrieval shares an encoder and concatenates absolute difference/product. This does not reproduce an upstream LM head or original LRA scores.

Training uses project AdamW/cosine/warmup/clipping/FP32; preserve A_log/D zero-decay flags. No output repair, test-based tuning, architecture/source substitution or silent fallback. Report source identity and environment.

The correction changes optimizer semantics and protocol hash. Do not resume old configurations/checkpoints under the new protocol. Old receipts remain evidence for their original revision only.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.

Explicit migration may fork a paused admitted31639c0 full-state checkpoint into a separate adaptive suite after new source/recovery/trace admission and a fresh native-worker zero-update restore/roundtrip check. Only revision/contract and controller metadata change; scientific configuration is locked. The fixed prefix and adaptive continuation retain lineage and must not be reported as an uninterrupted fixed-policy baseline. Original checkpoints are preserved; native migration remains executor validation.

Separately opted-in dynamic logical32/64/128 and physical1..128 policy uses committed samples for LR/progress and samples/s for throughput decisions. It changes optimizer-update count and training trajectory; it is not fixed-batch/common-protocol performance equivalence. FP32, model/data identity and total epoch/sample budget remain declared. Explicit migration from fixed31639c0 or adaptive96e542d preserves parameters, optimizer moments/counters, RNG, sampler, best/history and lineage in a separate suite, anchors LR at the source position, and requires new native admission. Controller timing resets. Native dynamic recovery, OOM behavior, accuracy and speed remain unverified until collaborator execution; owner checks use stdlib/mocks only.

FP32 performance opt-in: bounded ID-prefix CPU lookahead for both workers; reusable pinned host staging for Torch only; first16-update numeric timing. Migration from9a147cd preserves sample clock/controller/optimizer. No AMP/TF32/compiler or batch-policy change; native recovery and throughput pending. See docs/baselines/fp32-performance.md.

Completed-result preservation: original source/config/result/checkpoint/trace-readiness bytes are hash-verified and retained, never relaunched or recertified as the new executor. Only remaining jobs use new recovery gates and workers. Final delivery declares each result source revision; no whole-suite single-revision equivalence claim.
