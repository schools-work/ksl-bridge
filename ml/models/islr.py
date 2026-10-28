"""Isolated sign recognition: encoder + attention pooling + ArcMargin cosine head (Guide Part IV §2.2)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .pose_encoder import PoseEncoder


class AttnPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        w = self.q(h).squeeze(-1).masked_fill(pad, float("-inf")).softmax(-1)
        return torch.einsum("bt,btd->bd", w, h)


class ArcMarginHead(nn.Module):
    def __init__(self, d: int, n_classes: int, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, z: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.linear(F.normalize(z), F.normalize(self.W))
        if y is None:
            return self.s * cos
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        onehot = F.one_hot(y, cos.shape[1]).bool()
        return self.s * torch.where(onehot, torch.cos(theta + self.m), cos)


class ISLR(nn.Module):
    def __init__(self, n_classes: int, **enc_kw: Any):
        super().__init__()
        self.enc = PoseEncoder(subsample=2, **enc_kw)
        self.pool = AttnPool(self.enc.d)
        self.proj = nn.Sequential(nn.Linear(self.enc.d, 256), nn.BatchNorm1d(256))
        self.head = ArcMarginHead(256, n_classes)

    def embed(self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h, p = self.enc(x, m, pad)
        return self.proj(self.pool(h, p))

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.head(self.embed(x, m, pad), y)


def open_set_decision(
    z: torch.Tensor, prototypes: torch.Tensor, tau_reject: float, tau_ambig: float
) -> list[dict[str, Any]]:
    """Cosine vs class prototypes → accept / ambiguous / reject (Guide Part IV §2.4)."""
    cos = F.linear(F.normalize(z), F.normalize(prototypes))
    top = cos.topk(min(3, cos.shape[1]), dim=-1)
    out = []
    for vals, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        if vals[0] < tau_reject:
            out.append({"decision": "reject", "top": list(zip(idx, vals, strict=True))})
        elif len(vals) > 1 and vals[0] - vals[1] < tau_ambig:
            out.append({"decision": "ambiguous", "top": list(zip(idx, vals, strict=True))})
        else:
            out.append({"decision": "accept", "top": list(zip(idx, vals, strict=True))})
    return out
