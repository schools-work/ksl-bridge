"""Log-Mel features + SpecAugment for ASR (Guide Part IV §9)."""

from __future__ import annotations

import torch


def logmel(wave: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=400, win_length=400, hop_length=160, n_mels=n_mels)
    x = mel(wave).clamp(min=1e-10).log()
    return ((x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-5)).T


def spec_augment(feats: torch.Tensor, F: int = 27, T: int = 40, n_freq: int = 2, n_time: int = 2) -> torch.Tensor:
    x = feats.clone()
    for _ in range(n_freq):
        f = int(torch.randint(0, F + 1, (1,)))
        f0 = int(torch.randint(0, max(1, x.shape[1] - f), (1,)))
        x[:, f0 : f0 + f] = 0
    for _ in range(n_time):
        t = int(torch.randint(0, min(T, max(1, x.shape[0] // 20)) + 1, (1,)))
        t0 = int(torch.randint(0, max(1, x.shape[0] - t), (1,)))
        x[t0 : t0 + t] = 0
    return x
