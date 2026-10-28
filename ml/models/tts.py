"""Our TTS (FastSpeech2-style acoustic model). Vocoder in vocoder.py; G2P in g2p_sw.py (Swahili)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class VariancePredictor(nn.Module):
    def __init__(self, d: int, k: int = 3, p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, k, padding=k // 2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(d, d, k, padding=k // 2),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.out = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Repeat each phoneme state by its (integer) duration in frames. h: (B,N,D), durations: (B,N)."""
    out = [torch.repeat_interleave(h[b], durations[b].clamp(min=0), dim=0) for b in range(h.shape[0])]
    return nn.utils.rnn.pad_sequence(out, batch_first=True)


class FastSpeech2(nn.Module):
    def __init__(self, n_phones: int, d: int = 256, layers: int = 4, heads: int = 2, n_mels: int = 80):
        super().__init__()
        self.emb = nn.Embedding(n_phones, d, padding_idx=0)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.duration, self.pitch, self.energy = VariancePredictor(d), VariancePredictor(d), VariancePredictor(d)
        self.pitch_emb, self.energy_emb = nn.Linear(1, d), nn.Linear(1, d)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True), layers
        )
        self.mel = nn.Linear(d, n_mels)

    def forward(self, phones: torch.Tensor, durations: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.encoder(self.emb(phones), src_key_padding_mask=phones.eq(0))
        log_d = self.duration(h)
        p, e = self.pitch(h), self.energy(h)
        h = h + self.pitch_emb(p[..., None]) + self.energy_emb(e[..., None])
        dur = durations if durations is not None else torch.clamp(torch.round(torch.exp(log_d) - 1), min=1).long()
        frames = length_regulate(h, dur)
        return {"mel": self.mel(self.decoder(frames)), "log_duration": log_d, "pitch": p, "energy": e}


class Synthesizer:
    def __init__(self, acoustic: FastSpeech2, vocoder: nn.Module, phone_ids: dict[str, int], sr: int = 22050):
        self.acoustic, self.vocoder, self.phone_ids, self.sr = acoustic.eval(), vocoder.eval(), phone_ids, sr

    @classmethod
    def from_checkpoint(cls, path: Path) -> Synthesizer:
        from .vocoder import HiFiGANGenerator

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ac = FastSpeech2(len(ckpt["phones"]) + 1)
        ac.load_state_dict(ckpt["acoustic"])
        voc = HiFiGANGenerator()
        voc.load_state_dict(ckpt["vocoder"])
        return cls(ac, voc, {p: i + 1 for i, p in enumerate(ckpt["phones"])}, ckpt.get("sr", 22050))

    @torch.no_grad()
    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        from .g2p_sw import g2p

        phones = [self.phone_ids.get(p, 0) for p in g2p(text, lang)]
        mel = self.acoustic(torch.tensor([phones]))["mel"]
        return self.vocoder(mel.transpose(1, 2)).squeeze().numpy(), self.sr
