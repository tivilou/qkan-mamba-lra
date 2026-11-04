# Transformer mechanism contract

Source primitive: PyTorch 2.7.1 TransformerEncoderLayer/TransformerEncoder from pinned Torch 2.7.1+cu126 mirror. This is a classic attention reference, not the LRA JAX model or a competitive author recipe reproduction.

Freeze batch_first=True, norm_first=True, GELU, feedforward width*4, four heads/layers, width128, sinusoidal position embedding and explicit key padding mask. Disable nested tensor. Final LayerNorm/masked pooling belong to the adapter. AdamW/cosine/warmup/clipping/FP32 and validation-only selection belong to the common project recipe.

Received Transformer/Retrieval six-update CUDA canary passed exact recovery and trace semantic/replay. The dropout=0 composition probe supplies no external upstream fixture or installed-module hash, so formal source certification remains not_verifiable.
