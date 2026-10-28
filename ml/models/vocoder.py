"""HiFi-GAN V2-size generator (mel → waveform)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, dilations: tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(ch, ch, k, dilation=d, padding=d * (k - 1) // 2) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            x = x + c(F.leaky_relu(x, 0.1))
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, n_mels: int = 80, ch: int = 128, rates: tuple[int, ...] = (8, 8, 2, 2)):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, ch, 7, padding=3)
        ups, res = [], []
        c = ch
        for r in rates:
            ups.append(nn.ConvTranspose1d(c, c // 2, r * 2, stride=r, padding=r // 2))
            c //= 2
            res.append(ResBlock(c))
        self.ups, self.res = nn.ModuleList(ups), nn.ModuleList(res)
        self.post = nn.Conv1d(c, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for up, rb in zip(self.ups, self.res, strict=True):
            x = rb(up(F.leaky_relu(x, 0.1)))
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))
