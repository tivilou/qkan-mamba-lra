"""QKAN-Mamba: Full model for sequence classification."""
import torch
import torch.nn as nn

from .qkan_mamba_block import QKANMambaBlock


class QKANMambaModel(nn.Module):
    """QKAN-Mamba model for sequence classification (LRA)."""

    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_layers: int = 6,
        n_classes: int = 10,
        qkan_latent_dim: int = 32,
        qkan_reps: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            QKANMambaBlock(
                d_model=d_model,
                qkan_latent_dim=qkan_latent_dim,
                qkan_reps=qkan_reps,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_input)
        Returns:
            (B, n_classes) logits
        """
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if self.pooling == "mean":
            h = h.mean(dim=1)
        elif self.pooling == "last":
            h = h[:, -1, :]
        return self.head(h)
