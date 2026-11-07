#!/usr/bin/env bash
set -euo pipefail
QKAN_ACCELERATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$QKAN_ACCELERATION_ROOT/scripts/migrate_performance.sh" --torch-acceleration "$@"
