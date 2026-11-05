# Transformer mechanism contract

Source primitive: PyTorch 2.7.1 TransformerEncoderLayer/TransformerEncoder from pinned Torch 2.7.1+cu126 mirror. This is a classic attention reference, not the LRA JAX model or a competitive author recipe reproduction.

Freeze batch_first=True, norm_first=True, GELU, feedforward width*4, four heads/layers, width128, sinusoidal position embedding and explicit key padding mask. Disable nested tensor. Final LayerNorm/masked pooling belong to the adapter. AdamW/cosine/warmup/clipping/FP32 and validation-only selection belong to the common project recipe.

Received Transformer/Retrieval six-update CUDA canary passed exact recovery and trace semantic/replay. The dropout=0 composition probe supplies no external upstream fixture or installed-module hash, so formal source certification remains not_verifiable.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.
