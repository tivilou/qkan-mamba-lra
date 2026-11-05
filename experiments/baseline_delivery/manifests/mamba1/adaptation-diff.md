# Mamba1 adaptation boundary

Keep the pinned upstream mixer equations/initialization. Declare task embeddings (pad=1), scalar pixel projection, project LayerNorm/residual/dropout stack, post-block padding mask, final masked mean pooling and classification head. Retrieval shares an encoder and concatenates absolute difference/product. This does not reproduce an upstream LM head or original LRA scores.

Training uses project AdamW/cosine/warmup/clipping/FP32; preserve A_log/D zero-decay flags. No output repair, test-based tuning, architecture/source substitution or silent fallback. Report source identity and environment.

The correction changes optimizer semantics and protocol hash. Do not resume old configurations/checkpoints under the new protocol. Old receipts remain evidence for their original revision only.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.
