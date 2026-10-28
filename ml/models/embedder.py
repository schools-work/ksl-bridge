"""Our sentence encoder for RAG: small Transformer + mean pooling, trained contrastively (InfoNCE)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SentenceEncoderModel(nn.Module):
    def __init__(self, vocab: int, d: int = 384, layers: int = 6, heads: int = 6, max_len: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids.eq(0)
        h = self.tf(self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device)), src_key_padding_mask=pad)
        h = h.masked_fill(pad[..., None], 0).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
        return F.normalize(h, dim=-1)


def info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    logits = q @ p.T / temperature
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


class SentenceEncoder:
    def __init__(self, model: SentenceEncoderModel, sp: Any, max_len: int = 256):
        self.model, self.sp, self.max_len = model.eval(), sp, max_len

    @classmethod
    def from_checkpoint(cls, path: Path) -> SentenceEncoder:
        import sentencepiece as spm

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = SentenceEncoderModel(ckpt["vocab"], **ckpt["kw"])
        m.load_state_dict(ckpt["state_dict"])
        return cls(m, spm.SentencePieceProcessor(model_proto=ckpt["spm_proto"]))

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        ids = [[i + 1 for i in self.sp.encode(t)][: self.max_len] for t in texts]  # 0 is padding
        L = max(len(i) for i in ids) or 1
        t = torch.zeros(len(ids), L, dtype=torch.long)
        for k, row in enumerate(ids):
            t[k, : len(row)] = torch.tensor(row)
        return self.model(t).numpy()
