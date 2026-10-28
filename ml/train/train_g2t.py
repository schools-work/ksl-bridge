"""Gloss → English seq2seq (annotated + template + back-translated pairs; Guide Part IV §6.3)."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ksl_core.vocab import Vocab
from ml.models.seq2seq import Seq2Seq

from .common import device, load_config, seed_everything
from .loop import train
from .seq2seq_data import PairDataset, collate, read_pairs


def run(cfg: dict, reverse: bool = False) -> None:  # type: ignore[type-arg]
    import sentencepiece as spm

    seed_everything(cfg["seed"])
    tr_pairs = [p for f in cfg["train_files"] for p in read_pairs(f)]
    va_pairs = read_pairs(cfg["val_file"])
    sp = spm.SentencePieceProcessor(model_file=cfg["spm_model"])
    gv = Vocab.build([g for gl, _ in tr_pairs for g in gl])
    tr = DataLoader(PairDataset(tr_pairs, gv.stoi, sp, reverse), cfg["batch_size"], shuffle=True, collate_fn=collate)
    va = DataLoader(PairDataset(va_pairs, gv.stoi, sp, reverse), cfg["batch_size"], collate_fn=collate)
    src_n, tgt_n = (sp.get_piece_size(), len(gv)) if reverse else (len(gv), sp.get_piece_size())
    model = Seq2Seq(src_n, tgt_n, **cfg["model"])
    dev = device()

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        src, tgt = b
        logits = m(src, tgt[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1), ignore_index=1, label_smoothing=0.1
        )

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        tot, n = 0.0, 0
        for src, tgt in va:
            logits = m(src.to(dev), tgt[:, :-1].to(dev))
            tot += float(
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1).to(dev), ignore_index=1)
            )
            n += 1
        return {"val_ce": tot / max(1, n)}

    best, ckpt = train(model, loss_fn, tr, val_fn, cfg["epochs"], lr=cfg["lr"], device=dev, run_name=cfg["experiment"])
    torch.save(
        {
            "state_dict": torch.load(ckpt, weights_only=True),
            "src_itos": gv.itos,
            "tgt_vocab": tgt_n,
            "kw": cfg["model"],
            "spm_proto": open(cfg["spm_model"], "rb").read(),
        },
        ckpt.with_suffix(".bundle.pt"),
    )
    print(f"best val CE {best:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/gloss2text.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
