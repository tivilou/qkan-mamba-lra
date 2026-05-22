#!/bin/bash
# Run all LRA experiments for a given model and seed range.
# Usage: bash scripts/run_all.sh [model] [seed_start] [seed_end]
#   model: qkan_mamba | transformer | s4 | mamba_only
#   Example: bash scripts/run_all.sh qkan_mamba 0 4

MODEL=${1:-qkan_mamba}
SEED_START=${2:-0}
SEED_END=${3:-4}

TASKS=("listops" "text" "retrieval" "image" "pathfinder")
CONFIG_DIR="experiments/lra/configs"

for TASK in "${TASKS[@]}"; do
    for SEED in $(seq $SEED_START $SEED_END); do
        echo "=== Running $TASK | model=$MODEL | seed=$SEED ==="
        python experiments/lra/train.py \
            --config "$CONFIG_DIR/$TASK.yaml" \
            --seed $SEED
    done
done

echo "All experiments complete."
