"""QKAN Gate: Quantum Kolmogorov-Arnold Network gate for Mamba enhancement.

Simplified from OPFA-DARUAN (sister project /home/project):
- No frequency band partitioning (not needed for NLP)
- No ontology concept embedding
- Retains multi-axis encoding and adaptive measurement
- Bottleneck design: d_model → latent_dim → d_model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_qc import StateVector, TorchGates


class QKANGate(nn.Module):
    """Quantum KAN gate for sequence modeling.

    Architecture: down(x) → quantum circuit → up → sigmoid
    The quantum circuit uses data re-uploading with multi-axis encoding.
    """

    def __init__(self, d_model: int, latent_dim: int = 32, reps: int = 4):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.reps = reps

        self.down = nn.Linear(d_model, latent_dim)
        self.up = nn.Linear(latent_dim, d_model)

        # Variational parameters
        self.theta = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(latent_dim, reps + 1, 2))
        )

        # Learnable frequencies (geometric init)
        self.w = nn.Parameter(
            torch.tensor([2.0 ** l for l in range(reps)])
        )

        # Multi-axis encoding: cycle through z, x, y
        self._axes = ["z", "x", "y"] * ((reps // 3) + 1)

        # Adaptive measurement weights
        self.measure_proj = nn.Linear(latent_dim, 3)

        # Post-activation
        self.postact_weight = nn.Parameter(torch.ones(latent_dim))
        self.postact_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model) or (B, L, d_model)
        Returns:
            gate values in [0, 1], same shape as x
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)
        else:
            B = x.shape[0]

        h = self.down(x)  # (B, latent_dim)

        # Quantum circuit
        psi = StateVector(
            h.shape[0], self.latent_dim,
            device=h.device, dtype=torch.complex64
        )
        psi.h()

        for l in range(self.reps):
            psi.rz(self.theta[:, l, 0])
            psi.ry(self.theta[:, l, 1])

            encoded = h * self.w[l]
            axis = self._axes[l]
            if axis == "z":
                gate = TorchGates.rz_gate(encoded, dtype=torch.complex64)
            elif axis == "x":
                gate = TorchGates.rx_gate(encoded, dtype=torch.complex64)
            else:
                gate = TorchGates.ry_gate(encoded, dtype=torch.complex64)
            psi.state = torch.einsum("mnbi,bin->bim", gate, psi.state)

        psi.rz(self.theta[:, self.reps, 0])
        psi.ry(self.theta[:, self.reps, 1])

        # Adaptive measurement
        weights = F.softmax(self.measure_proj(h), dim=-1)
        state_saved = psi.state.clone()
        m_z = psi.measure_z()
        psi.state = state_saved.clone()
        m_x = psi.measure_x()
        psi.state = state_saved
        m_y = psi.measure_y()

        postacts = (
            weights[:, 0:1] * m_x +
            weights[:, 1:2] * m_y +
            weights[:, 2:3] * m_z
        )

        out = postacts * self.postact_weight + self.postact_bias
        gate_val = torch.sigmoid(self.up(out))

        return gate_val.reshape(orig_shape)
