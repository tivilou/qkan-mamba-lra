#!/usr/bin/env bash
# Explicit formal continuation; all assets and authenticated return are local.
set -euo pipefail
QKAN_FORMAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$QKAN_FORMAL_ROOT"
[[ -x .venv-baselines/bin/python ]] || { printf 'Missing local .venv-baselines/bin/python\n' >&2; exit 2; }
exec .venv-baselines/bin/python -B -m experiments.baseline_delivery.key_bias_formal "$@"
