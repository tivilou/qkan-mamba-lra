"""QKAN-Mamba Block: QKAN-gated Mamba for long-range sequence modeling."""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

import sys
sys.path.insert(0, '..')
from qkan.qkan_gate import QKANGate


class QKANMambaBlock(nn.Module):
    """QKAN-gated Mamba block.

    Architecture: y = x + QKAN_gate(x) * Mamba(x)
    """

    def __init__(
        self,
        d_model: int,
        qkan_latent_dim: int = 32,
        qkan_reps: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.qkan_gate = QKANGate(d_model, qkan_latent_dim, qkan_reps)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
            )
        else:
            raise ImportError("mamba-ssm is required")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        h = self.norm(x)
        g = self.qkan_gate(h)
        m = self.mamba(h)
        return x + g * m
