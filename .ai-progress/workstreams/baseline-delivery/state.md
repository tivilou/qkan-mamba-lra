# 基线复现与可验证实验交付

## Goal

- Deliver source-pinned, tested baseline runners to the collaborator, with reproducibility and return-artifact contracts.

## Current State

- One-command candidate implemented: `scripts/run_baselines.sh`, `experiments/baseline_delivery/`, handoff `docs/baselines/one-click.md`. Preparation, independent GPU jobs, full-state resume and immutable chunked return are implemented; formal baseline is not yet certified.
- Latest blocker: collaborator advanced to S4D/ListOps, where independent Conv1d also failed despite free VRAM. Installed cuDNN metadata=9.5.1.17, actual runtime=(9,26,0), loaded from this venv's nvidia/cudnn/lib even after removing loader environment variables. At `04d0f61`, core checks actual cuDNN tuple and repairs all pinned NVIDIA library files via force-reinstall from verified cache; selected GPUs now get a small Conv1d forward/backward check before model canaries. Collaborator repair/rerun pending.
- Data verification fixed at `36278f4`: collaborator subsequently verified all 15 splits' exact internal NPY identities despite different ZIP encoding. Source-bound member hashes preserve strict integrity; workers/resume bind actual local NPZ bytes. Collaborator Mamba 2.2.4 compiled/installed and bootstrap passed before reaching GPU preflight.
- Trace replay fix at `96fcb66`: collaborator Transformer/Retrieval GPU canary reached trace capture, then CPU NumPy pooling versus GPU FP32 pooling failed on 4/256 values (maximum absolute difference about 3.02e-6). Replay now independently decodes saved encoder/mask arrays and reconstructs pooling/pair/head with original device, dtype and batch shape. Tolerances remain rtol=1e-5, atol=1e-6; collaborator GPU rerun is still required.
- Dependency mirror at `f1adc22`: 44 official-source dependency artifacts (3,036,418,350 bytes), including cu126 Torch/CUDA wheels and pinned Mamba source, published to authenticated exchange in 64 MiB chunks. Complete HTTPS readback hashes pass. Installer now uses only this manifest and `pip --no-index`; Mamba source identity is certified by the pinned archive hash. Linux x86_64 / Python 3.11 supported; collaborator confirms 3.11.11. Data still comes directly from the author.
- Installer repair at `410b28c`: constrained cu126 Torch/NumPy/einops/Triton/Transformers/setuptools; active Mamba source changed to author v2.2.4 commit `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`, still Mamba1. Original surveyed pin remains historical evidence. Exchange configuration is checked before installation; final stack/source/extension imports are checked before data work. GPU installation remains unverified.
- Collaborator setup guide: `docs/baselines/collaborator-beginner-guide.zh-CN.md`, linked from README and one-click handoff. It gives direct Chinese steps from server setup through automatic data preparation and formal result upload; keep introductions, audit prose and troubleshooting out of this guide. Mamba CUDA extension opt-in is set in the environment steps.
- Engineering candidates: Transformer, explicitly pedagogical minimal upstream S4D, pinned Mamba1. Formal comparisons remain gated by foundation D01, cohort freeze and per-model fidelity.
- Primary-source inventory pins S4/S5/MEGA/Mamba and one explicitly unofficial LRU implementation. Candidate shortlist is provisional, not a current top-x claim.
- Legacy runner defects and per-model fidelity/recovery/trace delivery gates are documented in docs/baselines/delivery-contract.md.

## In Progress

- Supporting R01: two pages of recent primary-paper search and six detailed papers screened; B2S6/strong Transformer recipes and newer gates must be considered. See docs/baselines/recent-screen.md.
- No model implementation or formal run has been certified; no performance result is claimed.
- Owner-side 3080Ti queue diagnostics passed on 2026-10-29: unit correctness, Transformer/S4D minimal fresh-process exact recovery, trace structural/semantic/replay, and full v3 split/dictionary identity checks. Evidence: `docs/evidence/baseline-runner-diagnostics-20261029.json`.
- Upstream Apache-2.0 source asset round-trip passed; data is acquired directly from the author and is not redistributed through our exchange. Handoff commit `fac5ae5` is pushed to GitHub branch `research/collaboration-foundation`.

## Next Actions

- Hand the branch to the collaborator; they run actual GPU preflight and return receipts. In parallel, finish dataset-use qualification, cohort freeze, and per-model fidelity certification before any formal launch.
- Collaborator updates the branch and reruns preflight with exchange variables set; the installer restores Torch 2.7.1+cu126 and replaces the previously built Mamba 2.3.2 package. Do not reuse that old wheel as the new pin. CPU checks do not certify GPU execution.

## Blockers

- Dataset qualification is insufficient solely on the remaining source/use evidence category; cohort and model fidelity remain uncertified. `release.json` is not_released; no formal-ready launch package.
- Do not switch to P4: the authorized 3080Ti node is online and was used for the owner-side queue diagnostics. Formal release remains blocked by data-use qualification, cohort, per-model fidelity, and collaborator GPU preflight.

## Validation

- Official source commits and README/paper metadata retrieved through the authorized server; compact source receipt committed with this stage.
- CUDA file repair: 14 regressions pass through task `999c5b66a07b41f4aa88bf30900c3b0e`; evidence `docs/evidence/baseline-cudnn-runtime-20261103.json`. Checks mock the mismatch/repair; actual collaborator GPU repair remains unverified.
- Replay fix at `96fcb66`: 13 regression tests pass via task `4d6f70bf1ab6463fa8cac227d5b3d3be`; CPU Transformer fresh-process recovery and trace semantic/replay pass via `96ba0347c4724259b362a9bc0d99d08f`. Evidence: `docs/evidence/baseline-trace-replay-20261103.json`. Actual collaborator GPU rerun remains pending.
- Content-identity fix: scheduler tasks `71692a78b4424472842c9df6f78f32b8` and `03e91287fa8941539d83fc7019d0855d` complete; 11 regressions plus 15 reference splits, 3 dictionaries and 15 configuration constructions pass. Evidence: `docs/evidence/baseline-data-content-20261103.json`. No GPU preflight or qualification promotion claimed.
- Mirror receipt: `docs/evidence/baseline-install-mirror-20261103.json`; all 44 objects readback-verified, shell syntax and 8 regressions pass via scheduler task `dc5229a54f5541ccb009af023e1efa78`. No actual GPU installation/compilation is certified.
- Installer shell syntax and four regression tests pass through shared queue task `b7c32cc747614dafad2c3f55fc12b624` at `410b28c`; evidence `docs/evidence/baseline-bootstrap-20261103.json`. Session: sessions/2026-11/20261103-bootstrap-repair.md.
- Owner-side CPU engineering validation passed; it is not GPU or performance evidence. Non-Git upstream asset publication and idempotent round-trip passed; datasets remain unpublished.
- At 52257f2, 31 qualification/data/pixel-source/exchange tests passed through the shared scheduler; all Image/Pathfinder pixels and labels exactly match pinned source. These tests do not certify baseline fidelity; no baseline performance is claimed.

## Relevant Decisions

- None.

## Plan Control

- Plan: plans/one-click/v001.json
- Plan Version: 1
- Plan Status: active
- Focus Step: F01
- Focus Status: blocked
- Completed Steps: I01, V01, X01, H01
- Next Action: 合作者在实际 GPU 节点执行 preflight 并回传收据；研究方继续完成数据资格、cohort 与逐模型 fidelity
- Blocker: 正式运行仍被数据用途资格、cohort 冻结、逐模型 fidelity 及合作者 GPU 预检阻止
- Last Checkpoint: 2026-10-29
