#!/usr/bin/env bash
set -euo pipefail
QKAN_ADAPTIVE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$QKAN_ADAPTIVE_ROOT/scripts/run_formal_readiness.sh" --adaptive-memory "$@"
