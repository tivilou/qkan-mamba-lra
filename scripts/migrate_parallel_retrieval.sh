#!/usr/bin/env bash
# Explicit measured Retrieval Graph route; retain other tasks' eager execution.
set -euo pipefail
QKAN_PARALLEL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$QKAN_PARALLEL_ROOT/scripts/migrate_torch_acceleration.sh" --parallel-retrieval-graph "$@"
