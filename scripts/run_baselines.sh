#!/usr/bin/env bash
set -euo pipefail
QKAN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QKAN_ROOT"
QKAN_PYTHON="${QKAN_PYTHON:-python3}"
QKAN_MODELS="transformer,s4d_minimal,mamba1"
QKAN_MODE="run"
QKAN_ARGS=("$@")
for ((i=0;i<${#QKAN_ARGS[@]};i++)); do
  case "${QKAN_ARGS[$i]}" in
    --models) QKAN_MODELS="${QKAN_ARGS[$((i+1))]}" ;;
    --models=*) QKAN_MODELS="${QKAN_ARGS[$i]#--models=}" ;;
    --mode) QKAN_MODE="${QKAN_ARGS[$((i+1))]}" ;;
    --mode=*) QKAN_MODE="${QKAN_ARGS[$i]#--mode=}" ;;
  esac
done
if [[ " ${QKAN_ARGS[*]} " == *' --help '* || " ${QKAN_ARGS[*]} " == *' -h '* ]]; then
  exec "$QKAN_PYTHON" -B -m experiments.baseline_delivery.runner --help
fi
if [[ "$QKAN_MODE" == run ]]; then
  "$QKAN_PYTHON" -B -c $'from experiments.baseline_delivery.runner import check_release\nimport sys\ntry: check_release()\nexcept ValueError as e: print(str(e),file=sys.stderr);sys.exit(2)'
fi
if [[ "$QKAN_MODE" == prepare || "$QKAN_MODE" == preflight || "$QKAN_MODE" == run ]]; then
  "$QKAN_PYTHON" -B -m experiments.baseline_delivery.bootstrap exchange
fi
if [[ ! -x .venv-baselines/bin/python ]]; then
  "$QKAN_PYTHON" -m venv .venv-baselines
fi
QKAN_ENV="$QKAN_ROOT/.venv-baselines/bin/python"
QKAN_CONSTRAINTS="$QKAN_ROOT/requirements-baseline-install.txt"
if [[ "$QKAN_MODE" != upload ]] && ! "$QKAN_ENV" -c 'import numpy; assert numpy.__version__=="2.2.6"' 2>/dev/null; then
  "$QKAN_ENV" -m pip install -c "$QKAN_CONSTRAINTS" 'numpy==2.2.6'
fi
if [[ "$QKAN_MODE" == run || "$QKAN_MODE" == preflight ]] && ! "$QKAN_ENV" -B -m experiments.baseline_delivery.bootstrap core 2>/dev/null; then
  "$QKAN_ENV" -m pip install -c "$QKAN_CONSTRAINTS" 'torch==2.7.1+cu126' --index-url https://download.pytorch.org/whl/cu126
  "$QKAN_ENV" -m pip install -c "$QKAN_CONSTRAINTS" 'numpy==2.2.6' 'einops==0.8.1'
fi
# Mamba has a compiled CUDA extension and is installed only when selected.
if [[ "$QKAN_MODE" == run || "$QKAN_MODE" == preflight ]] && [[ ",${QKAN_MODELS}," == *,mamba1,* ]]; then
  export MAMBA_KEEP_CUDA_BUILD=TRUE
  if ! "$QKAN_ENV" -B -m experiments.baseline_delivery.bootstrap mamba 2>/dev/null; then
    command -v nvcc >/dev/null || { echo 'Mamba1 requires the matching CUDA toolkit (nvcc); no fallback model is substituted.' >&2; exit 2; }
    "$QKAN_ENV" -m pip install -c "$QKAN_CONSTRAINTS" packaging ninja 'setuptools==80.9.0' wheel
    "$QKAN_ENV" -m pip install -c "$QKAN_CONSTRAINTS" --no-build-isolation 'git+https://github.com/state-spaces/mamba.git@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9'
  fi
  "$QKAN_ENV" -B -m experiments.baseline_delivery.bootstrap mamba
fi
if [[ "$QKAN_MODE" == run || "$QKAN_MODE" == preflight ]]; then
  "$QKAN_ENV" -B -m experiments.baseline_delivery.bootstrap core
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$QKAN_ENV" -B -m experiments.baseline_delivery.runner "$@"
