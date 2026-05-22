"""Mamba-only baseline for LRA benchmark (no QKAN gate)."""
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required")
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaOnlyModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_classes: int = 10,
        pooling: str = "mean",
    ):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if self.pooling == "mean":
            h = h.mean(dim=1)
        elif self.pooling == "last":
            h = h[:, -1, :]
        return self.head(h)
