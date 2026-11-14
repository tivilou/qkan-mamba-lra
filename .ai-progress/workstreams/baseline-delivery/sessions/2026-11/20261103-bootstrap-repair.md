# Baseline installer dependency repair

- Date: 2026-11-03
- Scope: baseline-delivery
- User authorized fixing, testing and publishing the installer after collaborator screenshots showed Torch drift and missing exchange variables.
- Root cause: the previously surveyed Mamba source requires a newer Triton and Mamba3 dependencies. Unconstrained installation replaced Torch 2.7.1 with 2.15.0 after building a CUDA extension. Pinning Torch alone would produce a dependency conflict.
- Changes: active runtime pin now author v2.2.4 commit `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`, retaining the Mamba1 architecture. All pip calls use explicit core constraints. Check exchange variables before installing; check exact CUDA channel, core versions, upstream identity and extension import afterwards. Preserve original source-collection receipts. Existing 2.3.2 wheel is not reused as new source.
- Verification: shared scheduler task `b7c32cc747614dafad2c3f55fc12b624`, attempt `9cb7512bb29a41f4943a37e21fec47b4`, complete at source `410b28c`; shell syntax and four regression tests pass. Compact evidence records hashes and limitations.
- Infrastructure: initial task `7dca8eaffb444f01b2ae65c0c294730f` experienced missing launch ACK. Confirmed no durable record, retried identical admitted attempt through scheduler RPC; terminal failure was a recipe path placeholder unsupported by runtime. Fixed recipe to project-relative path and submitted a distinct task after terminal proof. No formal training or GPU used.
- Remaining: collaborator updates branch, configures exchange variables, reinstalls via preflight and verifies actual GPU import/forward/recovery. No GPU installation acceptance is claimed; formal release remains blocked.
- Git snapshot: started clean at `30a137b`, implementation committed as `410b28c`; this record and compact evidence are a follow-up commit. Preserve normal repository author identity.
