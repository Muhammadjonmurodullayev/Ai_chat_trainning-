"""SwiGLU Feed-Forward Network — Shazeer 2020."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN: (SiLU(xW_gate) * xW_up) W_down"""

    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        inner_dim = int(ff_dim * 2 / 3)
        self.w_gate = nn.Linear(embed_dim, inner_dim, bias=False)
        self.w_up = nn.Linear(embed_dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
