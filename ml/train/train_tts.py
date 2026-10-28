"""FastSpeech2 acoustic model: mel L1 + duration/pitch/energy MSE. Durations from a forced-alignment file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml.models.g2p_sw import g2p
from ml.models.tts import FastSpeech2

from .common import device, load_config, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description="rows: {text, lang, mel_npy, durations, pitch, energy}")
    ap.add_argument("--config", default="ml/configs/tts_fastspeech2.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    rows = [json.loads(x) for x in Path(cfg["train_jsonl"]).read_text().splitlines() if x.strip()]
    phones = sorted({p for r in rows for p in g2p(r["text"], r["lang"])})
    pid = {p: i + 1 for i, p in enumerate(phones)}
    model = FastSpeech2(len(phones) + 1, **cfg["model"]).to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for ep in range(cfg["epochs"]):
        tot = 0.0
        for r in rows:
            ph = torch.tensor([[pid[p] for p in g2p(r["text"], r["lang"])]], device=device())
            dur = torch.tensor([r["durations"]], device=device())
            mel = torch.from_numpy(np.load(r["mel_npy"])).float()[None].to(device())
            out = model(ph, dur)
            T = min(out["mel"].shape[1], mel.shape[1])
            loss = (
                F.l1_loss(out["mel"][:, :T], mel[:, :T])
                + F.mse_loss(out["log_duration"], torch.log(dur.float() + 1))
                + F.mse_loss(out["pitch"], torch.tensor([r["pitch"]], device=device()).float())
                + F.mse_loss(out["energy"], torch.tensor([r["energy"]], device=device()).float())
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        print(f"epoch {ep}: loss {tot / max(1, len(rows)):.4f}")
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"acoustic": model.state_dict(), "phones": phones}, cfg["out"])


if __name__ == "__main__":
    main()
