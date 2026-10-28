"""Masked Pose Modelling: self-supervised pre-training (Han et al. 2021; Guide Part IV §2.3)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .pose_encoder import PoseEncoder


class MaskedPoseModel(nn.Module):
    def __init__(self, d_feat: int = 892, mask_ratio: float = 0.4, span: int = 6, **enc_kw: Any):
        super().__init__()
        self.enc = PoseEncoder(d_feat=d_feat, subsample=1, **enc_kw)
        self.mask_token = nn.Parameter(torch.zeros(d_feat))
        self.dec = nn.Linear(self.enc.d, d_feat)
        self.ratio, self.span = mask_ratio, span

    def span_mask(self, B: int, T: int, device: torch.device | str) -> torch.Tensor:
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        n_spans = max(1, int(T * self.ratio / self.span))
        starts = torch.randint(0, max(1, T - self.span), (B, n_spans), device=device)
        for k in range(self.span):
            m.scatter_(1, (starts + k).clamp(max=T - 1), True)
        return m

    def forward(self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        tm = self.span_mask(B, T, x.device) & ~pad
        x_in = torch.where(tm[..., None], self.mask_token.expand_as(x), x)
        m_in = m & ~tm[..., None]
        h, _ = self.enc(x_in, m_in, pad)
        rec = self.dec(h)
        valid = tm[..., None] & m
        return (torch.abs(rec - x) * valid).sum() / valid.sum().clamp(min=1)
