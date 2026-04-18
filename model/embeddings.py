"""Token Embedding — Maps token IDs to dense vectors."""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Standard token embedding with sqrt(d) scaling."""

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self._scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self._scale
