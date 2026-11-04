# Declared native adapter boundary

- Official pinned source is imported with repository/commit/file/clean-tree checks or exact full-inventory offline snapshot checks.
- Five qualified task inputs, project pixel scaling x/255-.5, masked mean pooling and shared Retrieval abs-difference/product linear head use the common protocol, not author scores.
- MEGA owns its original sparse/real-number embedding, bidirectional EMA/gated attention, prenorm LayerNorm and source dropout. Clear derived EMA caches after loading weights; active operator unchanged. No unneeded outer norm.
- S5 retains native JAX complex state, source HiPPO initialization, conjugate symmetry, bidirectional scan, ZOH, dt=.001-.1, half_glu1 and prenorm LayerNorm. Token embedding precedes the official Dense encoder. LayerNorm is explicit; no mutable BatchNorm state. Modern JAX Array aliases the obsolete DeviceArray annotation only.
- Native S5 uses Optax standard SSM groups (B/Lambda/log_step/norm Adam no-decay; remaining AdamW), common warmup/cosine, gradient accumulation/clipping, saved explicit dropout key and NumPy sampler state. AMP disabled.
- The project classification Dense head explicitly uses JAX HIGHEST dot precision; the independent source-composition reference and compiled batched trace replay use the same dot precision. The pinned S5 encoder/scan is unchanged. This replaces an unspecified head precision choice; it is not an author precision-equivalence claim. Replay still independently reconstructs masked pooling from decoded encoders and checks pooled/logits at rtol=1e-5, atol=1e-6.
- MEGA uses independent CPython3.10/Torch1.13.1-cu117; S5 JAX0.4.23/Flax0.8.2 CUDA12.2/cuDNN8.9.7. Principal dependencies pinned, transitive freeze recorded and compared on reuse. This environment adaptation is not author-environment equivalence.
- Runtime source-composition forward/loss/all-gradient/one-update, all-task fresh-process recovery and complete trace/replay gates are mandatory before local admission. No current receipt is inferred from code presence.
- Forbidden: generic approximate replacement, silent padding/direction changes, test-based selection or claiming author result reproduction.

- Source and complete Linux cp310 dependency closure arrive through authenticated exchange with SHA-256 manifests. Use local Python3.10/venv without ensurepip; pip bootstrap and installation are offline. No public Conda/pip/Git fallback. This transport adaptation changes no active model operator.
