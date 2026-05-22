# QKAN-Mamba Research Proposal

## Title
QKAN-Mamba: Quantum Kolmogorov-Arnold Gated State Space Models for Long-Range Sequence Understanding

## Abstract (Draft)
We propose QKAN-Mamba, a hybrid architecture that enhances Mamba's selective state space mechanism with quantum circuit-based non-linear gating. While Mamba achieves linear-time long-range modeling via its selective scan, its internal gating relies on simple linear projections. We replace this with a Quantum KAN (Kolmogorov-Arnold Network) gate that provides richer feature interactions through parameterized quantum circuits. On the Long Range Arena benchmark, QKAN-Mamba achieves [TBD]% average accuracy, outperforming both vanilla Mamba and Transformer baselines while maintaining O(L) complexity.

## Motivation

### Why Mamba + QKAN?

Mamba's selective state space model has two components:
1. **Temporal modeling**: Selective scan propagates information over long sequences in O(L)
2. **Gating**: Linear projections + SiLU decide what information to keep/discard

The gating mechanism is Mamba's "brain" — it decides what's important. But it's surprisingly simple (just linear + activation). Our hypothesis: replacing this with a quantum circuit gate that can model complex feature interactions will improve Mamba's selectivity, especially on tasks requiring rich feature combinations.

### Evidence from Sister Project

In `/home/project` (opfa-daruan-clinical), we demonstrated:
- Quantum gates outperform classical gates by +5-15% on 4 clinical datasets
- The quantum circuit's non-linearity (via data re-uploading) provides modeling power that MLP gates cannot match
- On longer sequences (W=72 vs W=24), quantum gating benefits MORE than classical gating (+4.2% vs -1.8%)

This last point is key: **quantum gating scales better with sequence length**. This makes it ideal for pairing with Mamba on long-sequence tasks.

## Method

### QKAN Gate

```python
class QKANGate(nn.Module):
    """Quantum KAN gate for Mamba enhancement."""
    def __init__(self, d_model, latent_dim=32, reps=4):
        self.down = nn.Linear(d_model, latent_dim)
        self.quantum_circuit = DARUAN(dim=latent_dim, reps=reps)
        self.up = nn.Linear(latent_dim, d_model)
    
    def forward(self, x):
        # x: (B*L, d_model)
        h = self.down(x)
        h = self.quantum_circuit(h)  # Non-linear via data re-uploading
        return sigmoid(self.up(h))
```

### QKAN-Mamba Block

```python
class QKANMambaBlock(nn.Module):
    def __init__(self, d_model, latent_dim=32, reps=4):
        self.norm = LayerNorm(d_model)
        self.qkan_gate = QKANGate(d_model, latent_dim, reps)
        self.mamba = Mamba(d_model)
    
    def forward(self, x):
        h = self.norm(x)
        g = self.qkan_gate(h)  # (B, L, d_model)
        m = self.mamba(h)       # (B, L, d_model)
        return x + g * m
```

### Key Design Decisions

1. **No frequency partitioning**: Unlike OPFA-DARUAN, we don't partition the circuit into bands. NLP doesn't need clinical interpretability, and partitioning hurts performance (-5.3% in sister project).

2. **Multi-axis encoding retained**: Using RX/RY/RZ rotations based on feature position provides richer non-linearity than RZ-only.

3. **Adaptive measurement retained**: Context-dependent measurement basis improves expressiveness (+0.9% in ablation).

4. **Bottleneck design**: d_model → latent_dim → d_model keeps parameter count low. The quantum circuit operates in a compressed space.

## Experiments

### Datasets: Long Range Arena (LRA)

| Task | Length | Classes | Train Size | Description |
|------|--------|---------|------------|-------------|
| ListOps | 2048 | 10 | 96K | Nested math operations |
| Text | 4096 | 2 | 25K | IMDB char-level sentiment |
| Retrieval | 4000 | 2 | 147K | Document similarity |
| Image | 1024 | 10 | 45K | Sequential CIFAR-10 |
| Pathfinder | 1024 | 2 | 160K | Path connectivity |

### Baselines

- Transformer (vanilla, with relative PE)
- S4 (Structured State Space)
- Mamba (vanilla)
- H3 (Hungry Hungry Hippos)
- RWKV (if applicable)

### Hyperparameters (starting point)

```yaml
model:
  d_model: 128
  n_layers: 6
  qkan_latent_dim: 32
  qkan_reps: 4
  
training:
  lr: 1e-3
  batch_size: 32
  epochs: 50
  scheduler: cosine
  weight_decay: 0.01
```

## Timeline

1. Week 1: Set up LRA data loading + training infrastructure
2. Week 2: Implement QKAN-Mamba block, run on ListOps (simplest task)
3. Week 3: Full LRA benchmark run
4. Week 4: Ablation experiments + analysis
5. Week 5: Paper writing

## References

- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Tay et al. (2021). "Long Range Arena: A Benchmark for Efficient Transformers"
- Gu et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces" (S4)
- Liu et al. (2024). "KAN: Kolmogorov-Arnold Networks"
- Sister project experiments: /home/project/docs/model_narrative_evolution.md
