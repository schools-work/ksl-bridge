"""Shared builders for pose-model training (ISLR, CSLR, MPM, fingerspelling)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from ksl_core.vocab import Vocab
from ml.data.datasets import PoseSeqDataset, collate


def loaders(
    splits: dict[str, list[dict[str, Any]]], vocab: Vocab, cfg: dict[str, Any], no_mirror: set[str] | None = None
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    tr = PoseSeqDataset(
        splits["train"], vocab, train=True, max_len=cfg.get("max_len", 384), no_mirror_glosses=no_mirror
    )
    va = PoseSeqDataset(splits["val"], vocab, train=False, max_len=cfg.get("max_len", 384))
    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 4)
    return (
        DataLoader(tr, bs, shuffle=True, collate_fn=collate, num_workers=nw, drop_last=True),
        DataLoader(va, bs, shuffle=False, collate_fn=collate, num_workers=nw),
    )


def load_encoder_init(model: torch.nn.Module, path: str | None) -> None:
    """Initialise model.enc from an MPM checkpoint (keys prefixed 'enc.')."""
    if not path:
        return
    sd = torch.load(path, map_location="cpu", weights_only=True)
    enc = {k[4:]: v for k, v in sd.items() if k.startswith("enc.") and not k.startswith("enc.sub")}
    missing, _unexpected = model.enc.load_state_dict(enc, strict=False)  # type: ignore[union-attr]
    print(f"loaded MPM encoder init ({len(enc)} tensors; missing {len(missing)})")
