#!/usr/bin/env bash
# Run only in the separate adaptive checkout after pausing the original suite.
set -euo pipefail
QKAN_MIGRATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QKAN_MIGRATION_ROOT"
QKAN_MIGRATION_SUITE="$QKAN_MIGRATION_ROOT/runs/adaptive-migrated-$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/run_adaptive_readiness.sh --output "$QKAN_MIGRATION_SUITE" --gates-only "$@"
for QKAN_MIGRATION_ARG in "$@"; do
    if [[ "$QKAN_MIGRATION_ARG" == --gates-only ]]; then
        printf 'Checks finished; --gates-only requested. No formal training started. Suite: %s\n' "$QKAN_MIGRATION_SUITE"
        exit 0
    fi
done
exec .venv-baselines/bin/python -B scripts/formal_progress.py \
    --resume "$QKAN_MIGRATION_SUITE" --display inplace --interval 5
