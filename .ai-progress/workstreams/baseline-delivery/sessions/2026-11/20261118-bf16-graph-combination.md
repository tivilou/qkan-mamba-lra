# BF16 + Graph diagnostic delivery

- Request: try stacking the current BF16 analytic K-bias policy and CUDA Graph for additional speed.
- Implemented: new isolated per-document BF16 forward/backward Graph runtime, latest paused formal-source binder, five supervised stages, numerical parity, exact fresh-process2vs1+1 recovery, ABBAx2 full-update timing, source invariance and compact automatic return. Unified `experiment.py diagnose bf16-graph`; no new shell wrapper, no change to current formal strategy/worker/Graph helper.
- Controls: 19 focused stdlib/fake checks; full guarded suite and literal package verification recorded in docs/evidence/bf16-graph-delivery-20261118.json. No owner framework imports, model/GPU work, dependency install or collaborator process action.
- Source: a356914 formal Suite key-bias-formal-20261117T141411Z-047505d9. Latest user screenshot shows Step15048/Epoch17/micro16/logical128/BF16 analytic eager/noGraph; screenshot observation only. New diagnostic must use actual latest source after safe pause; never remigrate old b07 to fit a test.
- Next: collaborator runs docs/baselines/bf16-graph.md, automatic authenticated compact report returns; audit parity/recovery/timing. Formal Graph capability and long-run convergence remain pending. Resume original Suite with integrated dashboard after the isolated diagnostic exits.
- Package roles: .runtime/bf16-graph-20261118/candidate,change.patch,verification.json,rollback.py. Publication tracked in package publication.json if successful.
