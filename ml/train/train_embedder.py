"""Contrastive (InfoNCE) training of our sentence encoder on (question, answer-chunk) pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.models.embedder import SentenceEncoderModel, info_nce

from .common import device, load_config, seed_everything


def main() -> None:
    import sentencepiece as spm

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/embedder.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    sp = spm.SentencePieceProcessor(model_file=cfg["spm_model"])
    pairs = [json.loads(line) for line in Path(cfg["pairs_jsonl"]).read_text().splitlines() if line.strip()]
    model = SentenceEncoderModel(sp.get_piece_size() + 1, **cfg["model"]).to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    def batchify(texts: list[str]) -> torch.Tensor:
        ids = [[i + 1 for i in sp.encode(t)][:256] for t in texts]
        t = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
        for k, r in enumerate(ids):
            t[k, : len(r)] = torch.tensor(r)
        return t.to(device())

    bs = cfg["batch_size"]
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(pairs)).tolist()
        tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            b = [pairs[j] for j in perm[i : i + bs]]
            loss = info_nce(model(batchify([x["query"] for x in b])), model(batchify([x["positive"] for x in b])))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"epoch {ep}: loss {tot / max(1, len(perm) // bs):.4f}")
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": sp.get_piece_size() + 1,
            "kw": cfg["model"],
            "spm_proto": open(cfg["spm_model"], "rb").read(),
        },
        out,
    )


if __name__ == "__main__":
    main()
