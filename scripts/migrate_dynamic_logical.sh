#!/usr/bin/env bash
set -euo pipefail
QKAN_DYNAMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$QKAN_DYNAMIC_ROOT/scripts/migrate_fixed_to_adaptive.sh" --dynamic-logical-batch "$@"
