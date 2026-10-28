"""Our ASR: Conformer-CTC over log-Mel (same ConformerBlock as the pose encoder) + CTC decoder + domain LM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..decoding.ctc_beam import CTCPrefixBeamSearch
from .pose_encoder import ConformerBlock, SinusoidalPE, Subsample


class ConformerCTC(nn.Module):
    def __init__(self, vocab: int, d_in: int = 80, d: int = 256, layers: int = 12, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.sub = Subsample(d, d, 4)
        self.pe = SinusoidalPE(d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, p=p) for _ in range(layers)])
        self.out = nn.Linear(d, vocab)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, pad = self.sub(self.inp(x), pad)
        h = self.pe(h)
        for b in self.blocks:
            h = b(h, pad)
        return self.out(h), pad


class ConformerCTCRecognizer:
    def __init__(self, model: ConformerCTC, itos: list[str], lm: Any = None, beam: int = 8):
        self.model, self.itos, self.lm, self.beam = model.eval(), itos, lm, beam

    @classmethod
    def from_checkpoint(cls, path: Path) -> ConformerCTCRecognizer:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ConformerCTC(len(ckpt["itos"]), **ckpt["kw"])
        m.load_state_dict(ckpt["state_dict"])
        return cls(m, ckpt["itos"])

    @torch.no_grad()
    def transcribe(self, pcm: np.ndarray, lang_hint: str | None) -> tuple[str, str, float]:
        from .asr_features import logmel

        feats = logmel(torch.from_numpy(pcm))[None]
        logits, _ = self.model(feats, torch.zeros(1, feats.shape[1], dtype=torch.bool))
        lp = logits.log_softmax(-1)[0].numpy()
        dec = CTCPrefixBeamSearch(self.itos, lm=self.lm, beam=self.beam, skip_ids=frozenset({1, 3, 4}))
        dec.step(lp)
        best = dec.nbest(1, final=True)
        pieces = best[0][0] if best else []
        text = "".join(pieces).replace("▁", " ").strip()
        conf = float(np.exp(lp.max(-1)).mean())
        return text, lang_hint or "en", conf
