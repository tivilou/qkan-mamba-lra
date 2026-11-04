# Consolidated collaborator launch

The current delivery implements all five model workers and one entry point. It has not been run on the collaborator GPU at this revision. Existing historical receipts cannot certify it.

Run in the existing `/root/projects/qkan-mamba-lra` terminal, retaining the already configured QKAN_CACHE and project exchange variables:

```bash
git pull --ff-only origin research/collaboration-foundation
bash scripts/run_formal_readiness.sh --cache "$QKAN_CACHE" --gpus 0
```

The second command prepares isolated MEGA/S5 Python3.10 environments, verifies pinned upstream Git checkouts or complete hash-inventoried offline snapshots and existing core/Mamba installation, performs source forward/loss/gradient/update probes, and runs two-epoch six-update train/validation-only recovery canaries for all five models on all five tasks on each selected GPU. These bounded checks may take time; setup output is in the printed suite's `setup.log`, worker output in its preflight directories. No test rows are used by the admission checks.

Only after every gate passes and compact evidence returns successfully does the same command admit and start 25 full-data seed42 jobs (100 epochs each). Main cohort: Transformer, Mamba1, MEGA, S5. Minimal S4D is auxiliary. Further seeds follow analysis of this complete-data first seed. The committed release remains closed until actual evidence; a local runtime release binds reviewed policy, code, protocol and per-model receipts. Passing imports alone never releases training.

Formal workers use latest full-state recovery independently of validation-best weights. S5 saves native parameters, Optax state/schedule counters, explicit dropout key, sampler/permutation/cursor, best weights and history; MEGA uses the shared Torch worker and clears official EMA eval caches after weight loading. Real input/intermediate traces remain private; compact trace validation returns automatically. Completion returns selected weights, full latest states (chunked if large), configuration, environment and metrics through authenticated SHA-256 receipts.

For interruption, use the exact new `Suite:` path printed by this command with `--resume`. Never reuse the historical baseline-20261103T115633Z-fa88b308 suite: the protocol/optimizer changed. On resume, finished canaries and jobs are verified and reused; latest checkpoints resume incomplete jobs. Changed code/environment/protocol is rejected.

`--gates-only` explicitly stops after gates. Its same suite can later resume with the default command. `--gpus auto` or an explicit comma-separated list enables independent-job scheduling and per-device admission. Default is the known GPU0. Do not duplicate an active suite on the same GPUs.

Admission returns `qkan-mamba/formal-readiness/<new suite ID>/gate-report.json` and `gate-evidence.json` before training. Formal completion returns the existing baseline-suite whitelist. Failed stages return a compact failure receipt with local log paths/hashes and a bounded, credential-redacted setup log through the private exchange. Raw model/data logs remain local. Retry upload directly with return_artifacts for the immutable receipt if the network fails; retraining is unnecessary.

Environment changes are explicit: MEGA uses Torch1.13.1-cu117/fairseq; S5 uses JAX0.4.23/Flax0.8.2 and pinned CUDA12.2/cuDNN8.9 libraries. Pip resolves exclusively from complete hash-pinned local wheelhouses and records an exact freeze, checked on reuse; these are reviewed installation recipes, not a claim of an already tested binary environment. Source Build/fixture incompatibility or a failed exact recovery gate stops the command and must be fixed; there is no promise that unexecuted code is bug-free.

Verified here: Python AST, JSON/manifest/provenance and policy hashes, shell syntax when Git Bash is available, source/API inspection and package metadata. Per user instruction, no model import, package install, test/training or GPU run was executed on the owner machine.


## Offline installation contract

Assume the collaborator cannot access external internet. Full official Conda URLs are external and cannot establish reachability. The current setup never calls Conda, apt, public pip indexes, external Git, or author-data download URLs. Owner preparation downloads and publishes the complete MEGA/S5 Linux x86_64 CPython3.10 dependency closure and exact upstream source archives to the existing authenticated exchange. The committed strong-offline-mirror.json binds each file, chunk and source member to size/SHA-256. Credentials stay in the existing terminal environment, outside Git. Large binaries stay in the exchange.

The already observed local /usr/bin/python3.10 is used with stdlib venv --without-pip. A verified local pip23.3.2 wheel bootstraps pip into each isolated environment without ensurepip. Every install then uses --no-index and a verified local wheelhouse, including MEGA native extension compilation. ANTLR4.8's pure-Python wheel is prepared on our side from its hash-verified source distribution. No package is installed on our side. Interrupted generated environments are preserved under ignored .incomplete identities; the active myconda and core environment are preserved.

Existing exact pinned clean Git checkouts remain reusable. Otherwise, the script extracts official commit archives fetched only from the exchange, verifies the entire member inventory and pinned API hashes, and checks imported modules originate there. These are exact source snapshots, not replacement implementations. Source-level fidelity, runtime recovery and trace gates remain mandatory.

Qualified task data must already exist in QKAN_CACHE, as on the collaborator machine. A local raw/lra.zip can rebuild the qualified data if required; absent data stop clearly. Restricted task data are not distributed through this dependency mirror. Local Python3.10/stdlib venv and the existing native build toolchain are prerequisites; this installer does not attempt online OS/toolchain repair.

First failure runs/formal-ready-20261104T112338Z-ddafd722 exposed missing ensurepip and a partial venv. Second failure runs/formal-ready-20261104T114512Z-0a205127 exposed Conda's inherited Aliyun channel mapping HTTP404. The offline path supersedes the interim online Conda workaround. Fresh suites are required after pulling these changes; never resume a suite from a previous revision.

This delivery is verified by static inspection and asset publication checks only. It does not claim successful collaborator GPU execution. The same entry command proceeds to formal training automatically only when every actual-device gate and compact evidence return succeeds.

## Offline CPython header supplement

The 55c8abf collaborator run reached MEGA native compilation but system Python.h was missing. The mirror now also contains the complete unmodified CPython3.10 amd64 include closure extracted from the hash-verified Ubuntu libpython3.10-dev package. The installer verifies and stages all headers under .bootstrap/python310-headers and prepends project-local include flags to the MEGA build. No apt, public fallback, Python runtime replacement or system directory mutation is used. Cached dependency downloads are reused; pull and launch a fresh suite as above. Current repair has static/asset verification, not successful runtime confirmation.

Owner mirror preparation: run scripts/prepare_python_headers.py after scripts/prepare_strong_offline_mirror.py to append the fixed development-header asset before binding policy hashes and handing off the manifest.
