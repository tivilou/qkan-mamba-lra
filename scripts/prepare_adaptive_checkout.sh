#!/usr/bin/env bash
# Prepare code only. No model import, GPU job or change to the active checkout.
set -euo pipefail
QKAN_ACTIVE_ROOT="$(git rev-parse --show-toplevel)"
git -C "$QKAN_ACTIVE_ROOT" fetch origin research/collaboration-foundation
QKAN_ADAPTIVE_REV="$(git -C "$QKAN_ACTIVE_ROOT" rev-parse FETCH_HEAD)"
QKAN_ADAPTIVE_TARGET="$(dirname -- "$QKAN_ACTIVE_ROOT")/qkan-mamba-lra-adaptive-${QKAN_ADAPTIVE_REV:0:12}"
if [[ -e "$QKAN_ADAPTIVE_TARGET" ]]; then
    [[ "$(git -C "$QKAN_ADAPTIVE_TARGET" rev-parse --show-toplevel)" == "$QKAN_ADAPTIVE_TARGET" ]]
    [[ "$(git -C "$QKAN_ADAPTIVE_TARGET" rev-parse HEAD)" == "$QKAN_ADAPTIVE_REV" ]]
    [[ -z "$(git -C "$QKAN_ADAPTIVE_TARGET" status --porcelain --untracked-files=no)" ]]
else
    git -C "$QKAN_ACTIVE_ROOT" worktree add --detach "$QKAN_ADAPTIVE_TARGET" "$QKAN_ADAPTIVE_REV"
fi
for QKAN_ASSET in .venv-baselines .venv-mega .venv-s5 .bootstrap; do
    [[ -d "$QKAN_ACTIVE_ROOT/$QKAN_ASSET" ]] || { printf 'Missing local prerequisite: %s\n' "$QKAN_ASSET" >&2; exit 2; }
    if [[ -e "$QKAN_ADAPTIVE_TARGET/$QKAN_ASSET" || -L "$QKAN_ADAPTIVE_TARGET/$QKAN_ASSET" ]]; then
        [[ "$(readlink -f "$QKAN_ADAPTIVE_TARGET/$QKAN_ASSET")" == "$(readlink -f "$QKAN_ACTIVE_ROOT/$QKAN_ASSET")" ]]
    else
        ln -s -- "$QKAN_ACTIVE_ROOT/$QKAN_ASSET" "$QKAN_ADAPTIVE_TARGET/$QKAN_ASSET"
    fi
done
printf 'Prepared separate executor: %s\nActive executor HEAD unchanged. No GPU job started.\n' "$QKAN_ADAPTIVE_TARGET"
printf 'When the selected GPU is available, run:\ncd %q\nbash scripts/run_adaptive_readiness.sh --cache "$QKAN_CACHE" --gpus 0 --gates-only\n' "$QKAN_ADAPTIVE_TARGET"
