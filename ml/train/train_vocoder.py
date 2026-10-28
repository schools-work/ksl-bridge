"""HiFi-GAN generator, stage 1: multi-resolution STFT + mel reconstruction (adversarial stage 2 adds MPD/MSD)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ml.models.vocoder import HiFiGANGenerator

from .common import device, load_config, seed_everything


def stft_loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=y.device)
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        win = torch.hann_window(n_fft, device=y.device)
        S = torch.stft(y, n_fft, hop, window=win, return_complex=True).abs()
        Sh = torch.stft(y_hat, n_fft, hop, window=win, return_complex=True).abs()
        loss = (
            loss
            + (S - Sh).norm() / S.norm().clamp(min=1e-6)
            + (torch.log(S + 1e-6) - torch.log(Sh + 1e-6)).abs().mean()
        )
    return loss


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description="rows: {mel_npy, wav}")
    ap.add_argument("--config", default="ml/configs/hifigan.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    rows = [json.loads(x) for x in Path(cfg["train_jsonl"]).read_text().splitlines() if x.strip()]
    gen = HiFiGANGenerator(**cfg["model"]).to(device())
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg["lr"], betas=(0.8, 0.99))
    for ep in range(cfg["epochs"]):
        tot = 0.0
        for r in rows:
            mel = torch.from_numpy(np.load(r["mel_npy"])).float().T[None].to(device())
            wav, _ = sf.read(r["wav"], dtype="float32")
            y = torch.from_numpy(wav)[None].to(device())
            y_hat = gen(mel).squeeze(1)
            T = min(y.shape[1], y_hat.shape[1])
            loss = stft_loss(y[:, :T], y_hat[:, :T])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"epoch {ep}: stft loss {tot / max(1, len(rows)):.4f}")
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), cfg["out"])


if __name__ == "__main__":
    main()
