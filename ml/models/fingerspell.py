"""Fingerspelling recogniser: small CTC model on dominant-hand features + lexicon-constrained decoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .pose_encoder import PoseEncoder

LETTERS = ["<blank>"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]
# dominant (right, after canonicalisation) hand: hand-local shape (705:768) + bones (828:888)
HAND_COLS = np.r_[705:768, 828:888]


class FingerspellCTC(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4, heads: int = 4, **kw: Any):
        super().__init__()
        self.enc = PoseEncoder(d_feat=len(HAND_COLS), d=d, layers=layers, heads=heads, subsample=2, **kw)
        self.out = nn.Linear(d, len(LETTERS))

    def forward(self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, pad = self.enc(x, m, pad)
        return self.out(h), pad


def select_hand_features(feats: np.ndarray) -> np.ndarray:
    return feats[:, HAND_COLS]


def lexicon_rescore(hyp: str, lexicon: list[str], max_dist: int = 2) -> str:
    """Snap a noisy letter sequence to the nearest venue lexicon entry (menu items, staff names)."""

    def dist(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]

    best = min(lexicon, key=lambda w: dist(hyp, w), default=None)
    return best if best is not None and dist(hyp, best) <= max_dist else hyp
