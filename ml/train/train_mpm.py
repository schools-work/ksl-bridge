"""Masked Pose Modelling pre-training (unlabelled signing, incl. pretrain_only sources)."""

from __future__ import annotations

import argparse

import torch

from ksl_core.vocab import Vocab
from ml.models.mpm import MaskedPoseModel

from . import registry
from .carbon import tracked
from .common import by_split, device, load_config, read_manifest, seed_everything
from .loop import train
from .pose_tasks import loaders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/mpm_pretrain.yaml")
    ap.add_argument("--no-registry", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    splits = by_split(read_manifest(cfg["manifest"]))
    for s in splits.values():
        for it in s:
            it.setdefault("glosses", [])
    tr, va = loaders(splits, Vocab.build([]), cfg)
    model = MaskedPoseModel(**cfg["model"])
    dev = device()

    def loss_fn(m: torch.nn.Module, b: list) -> torch.Tensor:  # type: ignore[type-arg]
        X, M, P, *_ = b
        return m(X, M, P)

    @torch.no_grad()
    def val_fn(m: torch.nn.Module) -> dict[str, float]:
        m.eval()
        losses = [float(m(X.to(dev), Mk.to(dev), P.to(dev))) for X, Mk, P, *_ in va]
        return {"recon_l1": sum(losses) / max(1, len(losses))}

    run = None
    if not a.no_registry:
        exp = registry.ensure_experiment("mpm", cfg["experiment"], cfg["hypothesis"], "RQ2")
        run = registry.start_run(exp, "mpm", cfg, cfg["seed"])
    (best, ckpt), fp = tracked(train)(
        model,
        loss_fn,
        tr,
        val_fn,
        cfg["epochs"],
        lr=cfg["lr"],
        device=dev,
        run_name=cfg["experiment"],
        log=(lambda k, v, s: registry.log_metric(run, k, v, s)) if run else None,
    )
    if run:
        registry.finish_run(run, True, fp, artifact_uri=str(ckpt))
    print(f"best recon L1 {best:.4f} → {ckpt}")


if __name__ == "__main__":
    main()
