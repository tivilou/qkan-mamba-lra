# Transformer mechanism contract

Source primitive: PyTorch 2.7.1 TransformerEncoderLayer/TransformerEncoder from pinned Torch 2.7.1+cu126 mirror. This is a classic attention reference, not the LRA JAX model or a competitive author recipe reproduction.

Freeze batch_first=True, norm_first=True, GELU, feedforward width*4, four heads/layers, width128, sinusoidal position embedding and explicit key padding mask. Disable nested tensor. Final LayerNorm/masked pooling belong to the adapter. AdamW/cosine/warmup/clipping/FP32 and validation-only selection belong to the common project recipe.

Received Transformer/Retrieval six-update CUDA canary passed exact recovery and trace semantic/replay. The dropout=0 composition probe supplies no external upstream fixture or installed-module hash, so formal source certification remains not_verifiable.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.

Explicit migration may fork a paused admitted31639c0 full-state checkpoint into a separate adaptive suite after new source/recovery/trace admission and a fresh native-worker zero-update restore/roundtrip check. Only revision/contract and controller metadata change; scientific configuration is locked. The fixed prefix and adaptive continuation retain lineage and must not be reported as an uninterrupted fixed-policy baseline. Original checkpoints are preserved; native migration remains executor validation.

Separately opted-in dynamic logical32/64/128 and physical1..128 policy uses committed samples for LR/progress and samples/s for throughput decisions. It changes optimizer-update count and training trajectory; it is not fixed-batch/common-protocol performance equivalence. FP32, model/data identity and total epoch/sample budget remain declared. Explicit migration from fixed31639c0 or adaptive96e542d preserves parameters, optimizer moments/counters, RNG, sampler, best/history and lineage in a separate suite, anchors LR at the source position, and requires new native admission. Controller timing resets. Native dynamic recovery, OOM behavior, accuracy and speed remain unverified until collaborator execution; owner checks use stdlib/mocks only.

FP32 performance opt-in: bounded ID-prefix CPU lookahead for both workers; reusable pinned host staging for Torch only; first16-update numeric timing. Migration from9a147cd preserves sample clock/controller/optimizer. No AMP/TF32/compiler or batch-policy change; native recovery and throughput pending. See docs/baselines/fp32-performance.md.
