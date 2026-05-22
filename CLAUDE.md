# QKAN-Mamba: Quantum Kolmogorov-Arnold Gated State Space Models

## Project Overview

This project implements QKAN-Mamba, a hybrid architecture combining Quantum KAN (Kolmogorov-Arnold Network) gating with Mamba (Selective State Space Model) for long-range sequence understanding.

**Paper Title**: "QKAN-Mamba: Quantum Kolmogorov-Arnold Gated State Space Models for Long-Range Sequence Understanding"

**Core Idea**: Use quantum circuit-based non-linear gating (QKAN) to enhance Mamba's selective state space mechanism on long-sequence tasks where Mamba excels.

## Background & Motivation

This project originates from a sister project (`/home/project` — opfa-daruan-clinical) where we discovered:

1. **QKAN gating is powerful**: Quantum circuit gates significantly outperform classical gates (Sigmoid, MLP) by +5-15% across 4 clinical datasets
2. **Mamba needs long sequences**: On short sequences (24-72 steps), Mamba provides no advantage over Conv1d. Mamba's selective SSM mechanism needs 512+ steps to show its strength.
3. **The combination makes sense when applied correctly**: QKAN provides rich feature-level non-linear interactions, Mamba provides efficient O(L) long-range temporal modeling.

## Architecture

```
Input: x ∈ (B, L, d_model)    L=1K-4K, d=128-256
         │
    ┌─────┴──────────┐
    │  QKAN Gate      │  Quantum circuit on embedding dimensions
    │  down → QC →    │  Decides "which feature interactions matter"
    │  up → sigmoid   │
    └─────┬──────────┘
         │ g ∈ (B, L, d_model)
         │
    ┌─────┴─────┐
    │   Mamba    │  Selective SSM for long-range temporal modeling
    │  Block     │  Decides "how to propagate info over time"
    └─────┬─────┘
         │ m ∈ (B, L, d_model)
         │
    y = x + g ⊙ m     (gated residual)
         │
    × N layers → Task Head
```

## Quantum Circuit (from sister project)

The quantum gate uses Data Re-Uploading Architecture (DARUAN):
- Real parameterized quantum circuits (not "quantum-inspired")
- Complex-valued state vectors, unitary gate operations, Born rule measurements
- Implemented in PyTorch for GPU autodiff compatibility
- Key operations: RX, RY, RZ rotations, Hadamard, adaptive measurement

Source code to reuse from `/home/project/qkan/src/qkan/daruan/torch_qc.py`:
- `StateVector` class: quantum state management
- `TorchGates` class: RX, RY, RZ, H, S, CNOT gates

## Target Benchmark: Long Range Arena (LRA)

5 tasks designed to test long-range dependency modeling:

| Task | Seq Length | Type | Metric |
|------|-----------|------|--------|
| ListOps | 2048 | Math expression parsing | Accuracy |
| Text (IMDB) | 4096 | Sentiment classification (char-level) | Accuracy |
| Retrieval | 4000 | Document matching | Accuracy |
| Image (CIFAR) | 1024 | Image classification (flattened) | Accuracy |
| Pathfinder | 1024 | Path detection | Accuracy |

Reference scores:
- Transformer: ~54% avg
- S4: ~81% avg
- Mamba: ~82% avg
- Target (QKAN-Mamba): >83% avg

## Experiment Plan

### Exp 1: LRA Benchmark (main results)
QKAN-Mamba vs Transformer, S4, Mamba, Linear on all 5 LRA tasks.

### Exp 2: Gate Ablation
- Mamba alone (no gate)
- Sigmoid gate + Mamba
- MLP gate + Mamba
- QKAN gate + Mamba (ours)

### Exp 3: Sequence Length Sensitivity
Test on Text task with lengths 1K / 2K / 4K / 8K.
Hypothesis: QKAN-Mamba advantage grows with sequence length.

### Exp 4: Parameter Efficiency
Fixed parameter budget: compare QKAN-Mamba vs wider Mamba vs deeper Mamba.

## QKAN Design (vs OPFA-DARUAN in sister project)

| Aspect | OPFA-DARUAN (clinical) | QKAN-Mamba (this project) |
|--------|----------------------|---------------------------|
| Frequency partitioning | Yes (3 bands) | No (unnecessary for NLP) |
| Ontology concept embedding | Yes | No (use position/layer-aware modulation) |
| Multi-axis encoding | Yes (RX/RY/RZ per band) | Yes (keep — provides rich non-linearity) |
| Adaptive measurement | Yes | Yes (keep — improves expressiveness) |
| latent_dim | 16 (extreme efficiency) | 32-64 (NLP needs more capacity) |
| Spectral certificates | Yes | No (not needed for NLP) |

## Tech Stack

- Python 3.10+
- PyTorch 2.x
- mamba-ssm (Mamba implementation)
- Datasets: LRA benchmark (available via HuggingFace or direct download)

## Directory Structure (planned)

```
src/
  qkan/
    torch_qc.py          — Quantum gates (from sister project)
    qkan_gate.py         — QKAN gate module
  mamba/
    qkan_mamba_block.py  — QKAN-gated Mamba block
    model.py             — Full model (N layers + head)
  baselines/
    transformer.py
    s4.py
    mamba_only.py
experiments/
  lra/
    data_loader.py       — LRA data loading
    train.py             — Training loop
    configs/             — Hyperparameter configs per task
results/
docs/
```

## Key Lessons from Sister Project

1. **Window/sequence length matters**: QKAN gate benefits more from longer sequences (W=72 >> W=24 in clinical experiments)
2. **Don't over-constrain**: Frequency partitioning and ontology routing hurt performance. Keep the quantum circuit flexible.
3. **Identity mixer can work**: In clinical experiments, QKAN gate alone (no mixer) achieved best results. But for long sequences, Mamba's temporal modeling IS needed.
4. **Multi-seed experiments**: Always run 3-5 seeds for statistical significance.
5. **Ablation direction**: Verify that each component contributes positively before claiming it in the paper.

## Git Workflow

- Main branch: `main`
- Feature branches: `feature/<name>`
- Commit style: imperative mood, concise, explain "why" not "what"
