"""Continuous sign recognition: streaming Conformer + CTC + intermediate CTC (Guide Part IV §3)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .pose_encoder import PoseEncoder


class CSLR(nn.Module):
    def __init__(self, vocab_size: int, inter_ctc_layer: int = 4, subsample: int = 4, **enc_kw: Any):
        super().__init__()
        self.enc = PoseEncoder(subsample=subsample, **enc_kw)
        self.ctc = nn.Linear(self.enc.d, vocab_size)
        self.inter_layer = inter_ctc_layer
        self.inter = nn.Linear(self.enc.d, vocab_size)

    def forward(
        self, x: torch.Tensor, m: torch.Tensor, pad: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        enc = self.enc
        h = enc.inp(torch.cat([x * m, m.float()], -1))
        h, pad = enc.sub(h, pad)
        h = enc.pe(h)
        inter_logits = None
        for i, blk in enumerate(enc.blocks):
            h = blk(h, pad, attn_mask)
            if i + 1 == self.inter_layer:
                inter_logits = self.inter(h)
        return self.ctc(h), inter_logits, pad

    @staticmethod
    def loss(
        logits: torch.Tensor,
        inter: torch.Tensor | None,
        pad: torch.Tensor,
        targets: torch.Tensor,
        tgt_lens: torch.Tensor,
        inter_w: float = 0.3,
    ) -> torch.Tensor:
        out_lens = (~pad).sum(1)
        tg = torch.cat([t[:l] for t, l in zip(targets, tgt_lens, strict=True)])

        def ctc(lg: torch.Tensor) -> torch.Tensor:
            lp = lg.float().log_softmax(-1).transpose(0, 1)
            return F.ctc_loss(lp, tg, out_lens, tgt_lens, blank=0, zero_infinity=True)

        main = ctc(logits)
        return main + inter_w * ctc(inter) if inter is not None else main
