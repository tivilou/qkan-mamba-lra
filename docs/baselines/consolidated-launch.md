# Consolidated collaborator launch

The current delivery implements all five model workers and one entry point. It has not been run on the collaborator GPU at this revision. Existing historical receipts cannot certify it.

Run in the existing `/root/projects/qkan-mamba-lra` terminal, retaining the already configured QKAN_CACHE and project exchange variables:

```bash
git pull --ff-only origin research/collaboration-foundation
bash scripts/run_formal_readiness.sh --cache "$QKAN_CACHE" --gpus 0
```

The second command prepares isolated MEGA/S5 Python3.10 environments, verifies pinned upstream checkouts and existing core/Mamba installation, performs source forward/loss/gradient/update probes, and runs two-epoch six-update train/validation-only recovery canaries for all five models on all five tasks on each selected GPU. These bounded checks may take time; setup output is in the printed suite's `setup.log`, worker output in its preflight directories. No test rows are used by the admission checks.

Only after every gate passes and compact evidence returns successfully does the same command admit and start 25 full-data seed42 jobs (100 epochs each). Main cohort: Transformer, Mamba1, MEGA, S5. Minimal S4D is auxiliary. Further seeds follow analysis of this complete-data first seed. The committed release remains closed until actual evidence; a local runtime release binds reviewed policy, code, protocol and per-model receipts. Passing imports alone never releases training.

Formal workers use latest full-state recovery independently of validation-best weights. S5 saves native parameters, Optax state/schedule counters, explicit dropout key, sampler/permutation/cursor, best weights and history; MEGA uses the shared Torch worker and clears official EMA eval caches after weight loading. Real input/intermediate traces remain private; compact trace validation returns automatically. Completion returns selected weights, full latest states (chunked if large), configuration, environment and metrics through authenticated SHA-256 receipts.

For interruption, use the exact new `Suite:` path printed by this command with `--resume`. Never reuse the historical baseline-20261103T115633Z-fa88b308 suite: the protocol/optimizer changed. On resume, finished canaries and jobs are verified and reused; latest checkpoints resume incomplete jobs. Changed code/environment/protocol is rejected.

`--gates-only` explicitly stops after gates. Its same suite can later resume with the default command. `--gpus auto` or an explicit comma-separated list enables independent-job scheduling and per-device admission. Default is the known GPU0. Do not duplicate an active suite on the same GPUs.

Admission returns `qkan-mamba/formal-readiness/<new suite ID>/gate-report.json` and `gate-evidence.json` before training. Formal completion returns the existing baseline-suite whitelist. Failed stages return a compact failure receipt with local log paths/hashes; raw logs are retained locally to avoid accidental input or credential disclosure. Retry upload directly with return_artifacts for the immutable receipt if the network fails; retraining is unnecessary.

Environment changes are explicit: MEGA uses Torch1.13.1-cu117/fairseq; S5 uses JAX0.4.23/Flax0.8.2 and pinned CUDA12.2/cuDNN8.9 libraries. Pip resolves residual dependencies and records an exact freeze, checked on reuse; these are reviewed installation recipes, not a claim of an already tested binary environment. Source Build/fixture incompatibility or a failed exact recovery gate stops the command and must be fixed; there is no promise that unexecuted code is bug-free.

Verified here: Python AST, JSON/manifest/provenance and policy hashes, shell syntax when Git Bash is available, source/API inspection and package metadata. Per user instruction, no model import, package install, test/training or GPU run was executed on the owner machine.
