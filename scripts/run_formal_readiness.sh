#!/usr/bin/env bash
set -euo pipefail
QKAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QKAN_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export QKAN_MEGA_SOURCE="$QKAN_ROOT/.bootstrap/strong-upstream/mega"
export QKAN_S5_SOURCE="$QKAN_ROOT/.bootstrap/strong-upstream/s5"
exec .venv-baselines/bin/python -B -m experiments.baseline_delivery.readiness "$@"
