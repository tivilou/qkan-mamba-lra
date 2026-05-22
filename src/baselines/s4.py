"""S4 (Structured State Space) baseline for LRA benchmark."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4Block(nn.Module):
    """Simplified S4 block using diagonal state space (S4D)."""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt = nn.Parameter(torch.rand(d_model) * 0.1 + 0.001)

        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        A = -torch.exp(self.A_log)
        dt = F.softplus(self.dt)

        dA = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(0))
        dB = self.B.unsqueeze(0) * dt.unsqueeze(-1).unsqueeze(0)

        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = h * dA + dB * x[:, t, :].unsqueeze(-1)
            y_t = (h * self.C.unsqueeze(0)).sum(dim=-1) + self.D * x[:, t, :]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = self.dropout(self.out_proj(y))
        return residual + y


class S4Model(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 64,
        n_classes: int = 10,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.embedding = nn.Linear(d_input, d_model)
        self.blocks = nn.ModuleList([
            S4Block(d_model, d_state, dropout)
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
