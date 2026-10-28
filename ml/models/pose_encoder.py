"""Shared Conformer pose encoder (Guide Part IV §2.1)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Subsample(nn.Module):
    def __init__(self, d_in: int, d: int, factor: int):
        super().__init__()
        layers: list[nn.Module] = []
        c = d_in
        for _ in range(int(math.log2(factor))):
            layers += [nn.Conv1d(c, d, kernel_size=5, stride=2, padding=2), nn.GELU()]
            c = d
        self.net = nn.Sequential(*layers) if layers else nn.Conv1d(d_in, d, 1)
        self.factor = factor

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        if self.factor > 1:
            pad = pad[:, :: self.factor][:, : y.shape[1]]
        return y, pad


class ConvModule(nn.Module):
    def __init__(self, d: int, k: int = 15, p: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.do = nn.Dropout(p)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(self.ln(x)), dim=-1)
        y = y.masked_fill(pad[..., None], 0.0).transpose(1, 2)
        y = F.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.do(self.pw2(y))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int = 4, p: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(p), nn.Linear(ff * d, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.conv = ConvModule(d, p=p)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff * d), nn.SiLU(), nn.Dropout(p), nn.Linear(ff * d, d))
        self.ln_out = nn.LayerNorm(d)
        self.do = nn.Dropout(p)

    def forward(self, x: torch.Tensor, pad: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        a, _ = self.att(h, h, h, key_padding_mask=pad, attn_mask=attn_mask, need_weights=False)
        x = x + self.do(a)
        x = self.conv(x, pad)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class SinusoidalPE(nn.Module):
    def __init__(self, d: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return x + self.pe[offset : offset + x.shape[1]]


def chunk_attn_mask(T: int, chunk: int, left_chunks: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Bool (T,T); True = NOT allowed (PyTorch MHA convention). Streaming: own chunk + `left_chunks` previous."""
    idx = torch.arange(T, device=device)
    c = idx // chunk
    allowed = (c[None, :] <= c[:, None]) & (c[None, :] >= c[:, None] - left_chunks)
    return ~allowed


class PoseEncoder(nn.Module):
    def __init__(
        self, d_feat: int = 892, d: int = 256, layers: int = 8, heads: int = 4, subsample: int = 4, p: float = 0.1
    ):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * d_feat, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(p))
        self.sub = Subsample(d, d, subsample)
        self.pe = SinusoidalPE(d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, p=p) for _ in range(layers)])
        self.d = d

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(torch.cat([x * m, m.float()], dim=-1))
        h, pad = self.sub(h, pad)
        h = self.pe(h)
        for blk in self.blocks:
            h = blk(h, pad, attn_mask)
        return h, pad
