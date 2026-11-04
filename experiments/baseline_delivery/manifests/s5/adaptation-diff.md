# S5 adaptation boundary

Use the official JAX/Flax implementation in an independent worker; do not emulate it with a similarly named PyTorch/S4D module. Task token encoding, padding lengths, pixel projection and common symmetric pair head require a declared thin adapter and independent same-input/weight comparisons. Project common-head results do not reproduce author-head benchmark scores.

Complete recovery must include Flax params, batch_stats/other mutable collections, full Optax multi-transform state and counters, JAX PRNG keys, host RNGs, permutation/next cursor, epoch accumulators and immutable best-validation state. Source dropout and vmap RNG splitting must remain explicit. BatchNorm/micro-batch and direction/padding behavior need direct checks.

Future sample-trace.v2 capture/replay must use actual saved representations with JAX device/dtype semantics; CPU NumPy recomputation is not automatically equivalent. Preserve source optimizer special groups and declare the common schedule separately from the original recipe. Runtime and all certification gates are pending.
