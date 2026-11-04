# S5 source-stage contract

Official commit 3c18fdb6b06414da35e77b94b9cd855f6a95ef17. strong-source-registry.json pins native Flax S5SSM, HiPPO/initialization helpers, SequenceLayer, StackedEncoderModel and training helpers. strong_sources verifies clean commit/file identities and imports actual modules. s5_encoder requires explicit eigenbasis, conjugate symmetry, direction, discretization, timescales, layer and normalization choices.

Keep complex-valued diagonal state, input/output matrices, bilinear/ZOH discretization and jax.lax.associative_scan. Preserve real-pair parameter representation and source parameter-group optimizer semantics. The official sequence model includes a Dense input encoder, normalization, residual/GLU/dropout choices; classification produces log-softmax, while the native Retrieval model uses its own four-feature MLP head. These differ from the project's common head and must not be silently mixed.

Status: source factory only. Native training/checkpoint/trace adapter and executable environment remain pending. Upstream CPU requirements specify JAX/JAXlib .3.5 and Flax .5.2, while the GPU file literally contains an unresolved jax[cuda]>=version line; it is not a reproducible GPU lock. Do not install it blindly or replace the source to accommodate a newer JAX API.
