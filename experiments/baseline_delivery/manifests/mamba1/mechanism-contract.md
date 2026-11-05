# Mamba1 mechanism contract

Pinned source: mamba_ssm/modules/mamba_simple.py:Mamba at 95d8aba8a8c75aedcaa6143713b11e745e7cd0d9 (v2.2.4). Independently retrieved SHA-256: a17e4c51b582dc0d4d690a649eba521cd0c1ee3dc8f0473a0967cdc9ec0874e3. Source receipt: docs/evidence/qualification-closeout-sources-20261104.json.

Keep input projection, causal convolution/SiLU, input-dependent delta/B/C, negative exponential A, D skip, selective scan, output projection and delta initialization. Full-sequence path has no inference cache. Freeze d_state=16, d_conv=4, expand=2, dt_rank=auto. Mamba2/3 are different mechanisms.

Author source lines 111/115 mark A_log/D with _no_weight_decay=True. The old optimizer ignored these flags; the correction assigns zero decay and adds independent group assertions. Received composition evidence used the same defective factory on both sides. The correction and installed-source hash assertion have no runtime receipt yet.

Project training/selection/checkpoint behavior is declared in protocol.json and worker.py, not an upstream language-model recipe. No EMA/AMP. GPU recovery uses a bounded FP32 tolerance; atomic kernels are not claimed bitwise deterministic. Missing source fixture/identity and corrected optimizer evidence remain explicit in baseline-manifest.json; status not_verifiable.

Opt-in adaptive-memory execution changes only physical microbatch partitioning under the same logical update budget. It defines a distinct numerical trajectory; it does not establish equivalence to the fixed-policy run or inherit its checkpoint admission. Source-pinned mixer equations and precision remain unchanged. See `docs/baselines/adaptive-memory.md` for rollback, throughput search, bounded recovery coverage and pending hardware validation.

Adaptive execution also uses a validated logical-batch staging cache, one CPU batch of prefetch, device-side per-update statistics, five-second advisory status writes and reusable GPU rollback buffers with CPU fallback (immutable candidate state for S5). Positive throughput gains have no percentage floor and require multiple confirmation windows. FP32 and the logical update budget are unchanged; device behavior and speedup remain unverified for this revision.
