# Fastest remaining baseline job first

- Status: active
- Date: 2026-11-28
- Authority: user requested tasks sorted fast to slow, then continue experiments.

Supersedes Nov25Retrieval-last priority. Only the private dispatcher manifest/status view and dashboard/control metadata change; pinned science workers, configs, results and checkpoints are untouched. Runtime costs come from verified all-pending report084622Z-49834a2b/SHA4201748c85d43fd8ae2eccc23306ef5d147e5eee448ba9c80f6b6b0076b6618f. Use remaining, not full-from-scratch time; Mamba Retrieval is historical same-job fused estimate.

`docs/baselines/runtime-order-20261128.json` binds original manifest and14config hashes. `task-order.json` binds that reviewed estimate file and records costs. Completed summaries are validated first; a valid result with stale failed/pending status is treated computed in this invocation, never relaunched. Unfinished unestimated jobs stop rather than receiving fake zero cost. Original validators independently read original scientific bytes.

Fixed initial ordering by reviewed remaining costs; no speculative dynamic reranking during a job. Ordinary subsequent resume retains persisted scheduling policy. Explicit `--task-order retrieval-last` or `manifest` restores earlier scheduling without rewinding checkpoints. Native queue launch remains collaborator-run, training plus original dashboard in one terminal.
