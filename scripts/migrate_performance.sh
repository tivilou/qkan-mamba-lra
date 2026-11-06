#!/usr/bin/env bash
set -euo pipefail
QKAN_PERFORMANCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$QKAN_PERFORMANCE_ROOT/scripts/migrate_dynamic_logical.sh" --performance "$@"
