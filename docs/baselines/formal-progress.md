# Formal progress display and checkpoint continuation

The display supervisor in scripts/formal_progress.py refreshes in place on an interactive terminal (including tmux), every 15 seconds by default. Redirected output automatically uses plain appended snapshots; --display append explicitly selects that mode. It displays total/computed/running/pending/failed/paused counts; physical GPU, job name and PID; epoch, batch, optimizer steps and update percentage; measured optimizer steps/second, process elapsed time, remaining training-update ETA and heartbeat age; and job names grouped by state. Final delivery is displayed separately from computation completion.

Interactive rendering uses the alternate screen and redraws the same rows, without terminal queries. Width is measured in terminal cells including Chinese characters; long job-name lists are clipped with an ellipsis and oversized frames show a resize hint, preventing wrap/scroll. Terminal size is re-read each frame. Cursor and previous screen are restored on normal exit and handled interruption. Full snapshots remain available using --display append.

## No running job after resuming

The original readiness entry rechecks offline environments and data before scheduling jobs. During that interval status.json still contains the previous paused/pending states. Those counts alone do not establish a stalled or failed parent. The display now identifies watch-only mode, whether a recent supervisor heartbeat and live parent PID are available, parent elapsed time, and setup/parent log byte counts and write ages. Updated supervisors also identify fixed setup markers and data-validation messages from this launch only. These are observed entry events, not worker progress or proof that a stage has completed. Old supervisors lack launch offsets, so a new watch-only monitor reports liveness/log metadata without guessing a stage from historical logs.

Watch-only never resumes a paused job. Do not repeatedly interrupt/relaunch a live parent to refresh the display. In another tmux window inspect the current private logs locally:

```bash
tail -n 30 /root/projects/qkan-mamba-lra/runs/formal-ready-20261105T042214Z-1e90315e/progress-parent.log /root/projects/qkan-mamba-lra/runs/formal-ready-20261105T042214Z-1e90315e/setup.log
```

No raw log text is added to the dashboard or automatic return. A displayed parent PID/liveness heartbeat is not evidence that optimizer updates have resumed; require a running job and a fresh worker heartbeat.

## Switch display without interrupting an active supervisor

The 8adcd7c screenshot confirms active training resumed (Transformer/ListOps Epoch4). Leave that training/supervisor window running. In tmux press Ctrl+B, release, then C to create a new window; run the commands below there. Watch-only reads existing files, never starts/stops training and never writes experiment state. Ctrl+C in this new watch window exits only the monitor. Ctrl+C in the original supervisor window still requests training pause.

```bash
cd /root/projects/qkan-mamba-lra
git fetch origin research/collaboration-foundation
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/formal_progress-new.py
.venv-baselines/bin/python -B .bootstrap/formal_progress-new.py --resume runs/formal-ready-20261105T042214Z-1e90315e --watch-only --display inplace --interval 5
```

Use fetch only; do not pull/change the pinned executor revision. The original supervisor may continue appending in its original window, but the new view refreshes in place without restarting training.

The supervisor reads existing structured worker heartbeats and frozen train-row counts. Rates need two advancing observations in the current process; unavailable rates/ETAs say pending measurement. ETA applies only to remaining optimizer updates of the active job, not all 25 jobs or validation/test/upload. A quiet heartbeat gets a caution after 120 seconds, never an automatic failure or kill. The original worker does not report distinct validation/save/trace phases; the UI explicitly states this limitation. No metrics, labels, data arrays, weights or optimizer state are modified by the monitor. Every child keeps its independent worker.log; parent runner output goes to progress-parent.log. The display's own outer heartbeat is explicitly marked as supervisor liveness, not worker progress. Launch argv and display SHA are recorded privately per attempt.

## Switch the admitted 31639c0 suite to the new display

The current runtime certificates, checkpoint contract and configuration pin executor revision 31639c014a62504f6596e75cf56a1abd5068bce4. Do **not** git pull in that checkout. A fetch can retrieve the independent display script without changing HEAD, the training code or the reviewed runtime policy. The local .bootstrap path is already ignored. This is a display upgrade, not cross-revision training migration.

In the existing qkan tmux window, press Ctrl+C **once** and wait for the original runner to save/checkpoint, clean up children and return to the shell prompt. Do not type the commands while training still owns the terminal. Do not kill the GPU worker directly. Then, from /root/projects/qkan-mamba-lra, run:

```bash
git fetch origin research/collaboration-foundation
git show FETCH_HEAD:scripts/formal_progress.py > .bootstrap/formal_progress.py
.venv-baselines/bin/python -B .bootstrap/formal_progress.py \
  --resume runs/formal-ready-20261105T042214Z-1e90315e
```

Run the commands in order; stop if fetch or extraction reports an error. The supervisor refuses a changed executor revision, dirty tracked sources, missing admission report, changed cache or a suite whose original parent still holds its lock. It then invokes the original consolidated entry with the original suite/cache/GPU settings. The original admission policy still checks all bindings. Existing setup/gates are verified/reused, completed jobs retained, and unfinished jobs resume their latest complete checkpoints. There is no new formal suite or fresh-training fallback for an existing checkpoint. No repeat dependency download is added.

The old runner's safe-pause grace is 90 seconds before process-group escalation; if an update takes longer, continuation uses the latest committed checkpoint and may repeat work since that checkpoint. Do not promise that all in-memory work is saved. Checkpoint generation/hash validation remains authoritative.

To pause the new supervisor, press Ctrl+C once. It forwards a safe-pause request to the original parent and waits for its existing checkpoint/cleanup protocol. Do not launch a duplicate suite. A nonzero exit, paused status or returned shell prompt is not completion; require delivery-complete.json.

For observation without restart or writes, the same display also supports:

```bash
.venv-baselines/bin/python -B .bootstrap/formal_progress.py \
  --resume runs/formal-ready-20261105T042214Z-1e90315e --watch-only
```

Owner validation is restricted to stdlib AST/argument parsing and temporary structured display fixtures (progress/rate/ETA/stale-heartbeat behavior). No model import, dependency installation, training or live collaborator interruption was performed. Linux supervision and actual checkpoint continuation are executor checks.
