#!/bin/bash
set -e

# Run LRA experiments for a selected model, seed range, and task list.
# Usage:
#   bash scripts/run_all.sh [model] [seed_start] [seed_end] [tasks]
#
# Examples:
#   bash scripts/run_all.sh qkan_mamba 0 4
#   bash scripts/run_all.sh transformer 0 0 text
#   bash scripts/run_all.sh mamba_only 0 2 text,image,pathfinder
#
# model: qkan_mamba | transformer | s4 | mamba_only
# tasks: all | listops | text | retrieval | image | pathfinder | comma-separated list

MODEL=${1:-qkan_mamba}
SEED_START=${2:-0}
SEED_END=${3:-4}
TASK_ARG=${4:-all}

CONFIG_DIR="experiments/lra/configs"
ALL_TASKS=("listops" "text" "retrieval" "image" "pathfinder")
VALID_MODELS=("qkan_mamba" "transformer" "s4" "mamba_only")

contains() {
    local value=$1
    shift
    for item in "$@"; do
        if [[ "$item" == "$value" ]]; then
            return 0
        fi
    done
    return 1
}

if ! contains "$MODEL" "${VALID_MODELS[@]}"; then
    echo "Unknown model: $MODEL"
    echo "Valid models: ${VALID_MODELS[*]}"
    exit 1
fi

if [[ "$TASK_ARG" == "all" ]]; then
    TASKS=("${ALL_TASKS[@]}")
else
    IFS=',' read -ra TASKS <<< "$TASK_ARG"
fi

for TASK in "${TASKS[@]}"; do
    if ! contains "$TASK" "${ALL_TASKS[@]}"; then
        echo "Unknown task: $TASK"
        echo "Valid tasks: all, ${ALL_TASKS[*]}"
        exit 1
    fi
done

for TASK in "${TASKS[@]}"; do
    for SEED in $(seq "$SEED_START" "$SEED_END"); do
        echo "=== Running $TASK | model=$MODEL | seed=$SEED ==="
        python experiments/lra/train.py \
            --config "$CONFIG_DIR/$TASK.yaml" \
            --model "$MODEL" \
            --seed "$SEED"
    done
done

echo "All experiments complete."
