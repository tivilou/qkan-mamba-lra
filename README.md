# QKAN-Mamba

Quantum Kolmogorov-Arnold Gated State Space Models for Long-Range Sequence Understanding.

## Overview

QKAN-Mamba combines quantum circuit-based non-linear gating (QKAN) with Mamba's selective state space mechanism for long-range sequence tasks. The quantum gate learns rich feature interactions while Mamba handles efficient temporal modeling.

## Architecture

```
Input (B, L, d_model) → QKAN Gate → Mamba Block → Gated Residual → Output
                         (feature)    (temporal)
```

## Setup

```bash
# Clone
git clone https://github.com/tivilou/qkan-mamba-lra.git
cd qkan-mamba-lra

# Install dependencies (requires CUDA)
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a single experiment
python experiments/lra/train.py --config experiments/lra/configs/listops.yaml

# Run all tasks with multiple seeds
bash scripts/run_all.sh qkan_mamba 0 4
```

## Project Structure

```
src/
  qkan/           Quantum KAN gate (circuit + gate module)
  mamba/          QKAN-Mamba block and full model
  baselines/      Transformer, S4, Mamba-only baselines
experiments/
  lra/
    train.py      Training loop
    data_loader.py  LRA data loading
    configs/      Per-task YAML configs
scripts/          Utility scripts
results/          Experiment outputs (JSON + checkpoints)
docs/             Documentation
```

## Benchmark: Long Range Arena (LRA)

| Task | Seq Length | Classes | Type |
|------|-----------|---------|------|
| ListOps | 2048 | 10 | Math expression |
| Text | 4096 | 2 | Sentiment (char) |
| Retrieval | 4000 | 2 | Document matching |
| Image | 1024 | 10 | CIFAR-10 (flat) |
| Pathfinder | 1024 | 2 | Path detection |

## License

Apache 2.0
