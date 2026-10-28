"""Fingerspelling CTC on dominant-hand features."""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ksl_core.normalize import build_features, normalize_sequence
from ml.decoding.ctc_beam import greedy_decode
from ml.eval.metrics import cer
from ml.models.fingerspell import LETTERS, FingerspellCTC, select_hand_features

from .common import by_split, device, load_config, read_manifest, seed_everything
from .loop import train

STOI = {c: i for i, c in enumerate(LETTERS)}


class FsDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, items: list[dict]):  # type: ignore[type-arg]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        it = self.items[i]
        f, m = build_features(
            normalize_sequence(
                np.load(it["landmarks_path"]).astype(np.float32), mirror=it.get("dominant_hand") == "left"
            )
        )
        word = it["word"].upper()
        return (
            torch.from_numpy(select_hand_features(f)),
            torch.from_numpy(select_hand_features(m)),
            torch.tensor([STOI[c] for c in word if c in STOI]),
        )


def collate(b: list) -> tuple[torch.Tensor, ...]:  # type: ignore[type-arg]
    xs, ms, ys = zip(*b, strict=True)
    T = max(x.shape[0] for x in xs)
    X = torch.zeros(len(xs), T, xs[0].shape[1])
    M = torch.zeros(len(xs), T, xs[0].shape[1], dtype=torch.bool)
    P = torch.ones(len(xs), T, dtype=torch.bool)
    for i, (x, m) in enumerate(zip(xs, ms, strict=True)):
        X[i, : len(x)], M[i, : len(x)], P[i, : len(x)] = x, m, False
    YL = torch.tensor([len(y) for y in ys])
    return X, M, P, torch.nn.utils.rnn.pad_sequence(list(ys), batch_first=True), YL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/fingerspell_ctc.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    splits = by_split(read_manifest(cfg["manifest"]))
    tr = DataLoader(FsDataset(splits["train"]), cfg["batch_size"], shuffle=True, collate_fn=collate, drop_last=True)
    va = DataLoader(FsDataset(splits["val"]), cfg["batch_size"], collate_fn=collate)
    model = FingerspellCTC(**cfg["model"])
    dev = device()

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        X, M, P, Y, YL = b
        logits, pad = m(X, M, P)
        lp = logits.float().log_softmax(-1).transpose(0, 1)
        tg = torch.cat([y[:l] for y, l in zip(Y, YL, strict=True)])
        return F.ctc_loss(lp, tg, (~pad).sum(1), YL, blank=0, zero_infinity=True)

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        refs, hyps = [], []
        for X, M, P, Y, YL in va:
            logits, pad = m(X.to(dev), M.to(dev), P.to(dev))
            lp = logits.log_softmax(-1).cpu().numpy()
            for b in range(len(Y)):
                hyps.append("".join(greedy_decode(lp[b, : int((~pad[b]).sum())], LETTERS)))
                refs.append("".join(LETTERS[i] for i in Y[b, : YL[b]].tolist()))
        return {"cer": cer(refs, hyps)}

    best, ckpt = train(model, loss_fn, tr, val_fn, cfg["epochs"], lr=cfg["lr"], device=dev, run_name=cfg["experiment"])
    print(f"best CER {best:.3f} → {ckpt}")


if __name__ == "__main__":
    main()
