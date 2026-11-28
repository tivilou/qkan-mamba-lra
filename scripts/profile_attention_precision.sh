#!/usr/bin/env bash
set -euo pipefail
QKAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QKAN_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec .venv-baselines/bin/python -B -m experiments.baseline_delivery.attention_precision_diagnostic "$@"
