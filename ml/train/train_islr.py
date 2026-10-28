"""Isolated sign recognition training (optionally MPM-initialised)."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ksl_core.vocab import Vocab
from ml.models.islr import ISLR

from . import registry
from .carbon import tracked
from .common import by_split, device, load_config, read_manifest, seed_everything
from .loop import train
from .pose_tasks import load_encoder_init, loaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/islr_transformer.yaml")
    ap.add_argument("--no-registry", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    splits = by_split(read_manifest(cfg["manifest"]))
    vocab = Vocab.build([g for it in splits["train"] for g in it["glosses"]])
    tr, va = loaders(splits, vocab, cfg, set(cfg.get("no_mirror_glosses", [])))
    model = ISLR(len(vocab), **cfg["model"])
    load_encoder_init(model, cfg.get("init_from_mpm"))
    dev = device()

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        X, M, P, _, Y, _ = b
        y = Y[:, 0]
        return F.cross_entropy(m(X, M, P, y), y, label_smoothing=cfg.get("label_smoothing", 0.1))

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        m.eval()
        correct = total = 0
        for X, M, P, _, Y, _ in va:
            pred = m(X.to(dev), M.to(dev), P.to(dev)).argmax(-1).cpu()
            correct += int((pred == Y[:, 0]).sum())
            total += len(Y)
        return {"top1": correct / max(1, total)}

    run = None
    if not a.no_registry:
        exp = registry.ensure_experiment("islr", cfg["experiment"], cfg["hypothesis"], "RQ2")
        run = registry.start_run(exp, "islr", cfg, cfg["seed"])
    (best, ckpt), fp = tracked(train)(
        model,
        loss_fn,
        tr,
        val_fn,
        cfg["epochs"],
        lr=cfg["lr"],
        device=dev,
        run_name=cfg["experiment"],
        lower_is_better=False,
        log=(lambda k, v, s: registry.log_metric(run, k, v, s)) if run else None,
    )
    torch.save(
        {"state_dict": torch.load(ckpt, weights_only=True), "itos": vocab.itos, "enc_kw": cfg["model"]},
        ckpt.with_suffix(".bundle.pt"),
    )
    if run:
        registry.finish_run(run, True, fp, artifact_uri=str(ckpt.with_suffix(".bundle.pt")))
    print(f"best val top-1 {best:.3f}")


if __name__ == "__main__":
    main()
