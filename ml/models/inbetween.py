"""Learned in-betweening (Guide Part IV §11.2): predict k transition frames between two signs."""

from __future__ import annotations

import torch
from torch import nn


class InBetweener(nn.Module):
    """Input: last frame of sign A, first frame of sign B, gloss ids → k frames (107×3 each)."""

    def __init__(self, n_glosses: int, k: int = 8, d: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        self.k = k
        self.pose_in = nn.Linear(107 * 3, d)
        self.gloss = nn.Embedding(n_glosses, d)
        self.queries = nn.Parameter(torch.randn(k, d) * 0.02)
        self.tf = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers
        )
        self.out = nn.Linear(d, 107 * 3)

    def forward(self, a_end: torch.Tensor, b_start: torch.Tensor, ga: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        B = a_end.shape[0]
        mem = torch.stack(
            [self.pose_in(a_end.flatten(1)) + self.gloss(ga), self.pose_in(b_start.flatten(1)) + self.gloss(gb)], dim=1
        )
        h = self.tf(self.queries.expand(B, -1, -1), mem)
        t = torch.linspace(0, 1, self.k + 2, device=a_end.device)[1:-1][None, :, None, None]
        linear = (1 - t) * a_end[:, None] + t * b_start[:, None]  # residual over linear interpolation
        return linear + self.out(h).view(B, self.k, 107, 3)
