"""Torch datasets + collate (features, presence masks, gloss targets)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ksl_core.normalize import build_features, normalize_sequence
from ksl_core.vocab import Vocab

from .augment import augment


class PoseSeqDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self,
        items: list[dict[str, Any]],
        vocab: Vocab,
        train: bool,
        max_len: int = 384,
        no_mirror_glosses: set[str] | None = None,
    ):
        self.items, self.vocab, self.train, self.max_len = items, vocab, train, max_len
        self.no_mirror = no_mirror_glosses or set()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        it = self.items[i]
        xyz = it["xyz"] if "xyz" in it else np.load(it["landmarks_path"])
        xyz = np.asarray(xyz, dtype=np.float32)
        if self.train:
            xyz = augment(xyz, allow_mirror=not (set(it["glosses"]) & self.no_mirror))
        x = normalize_sequence(xyz, mirror=(it.get("dominant_hand") == "left"))
        feats, mask = build_features(x)
        if feats.shape[0] > self.max_len:
            idx = np.linspace(0, feats.shape[0] - 1, self.max_len).astype(int)
            feats, mask = feats[idx], mask[idx]
        return torch.from_numpy(feats), torch.from_numpy(mask), torch.tensor(self.vocab.encode(it["glosses"]))


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    feats, masks, targets = zip(*batch, strict=True)
    lens = torch.tensor([f.shape[0] for f in feats])
    T, F = int(lens.max()), feats[0].shape[1]
    X = torch.zeros(len(feats), T, F)
    M = torch.zeros(len(feats), T, F, dtype=torch.bool)
    pad = torch.ones(len(feats), T, dtype=torch.bool)
    for i, (f, m) in enumerate(zip(feats, masks, strict=True)):
        X[i, : f.shape[0]] = f
        M[i, : f.shape[0]] = m
        pad[i, : f.shape[0]] = False
    tgt_lens = torch.tensor([len(t) for t in targets])
    Y = torch.nn.utils.rnn.pad_sequence(list(targets), batch_first=True, padding_value=-100)
    return X, M, pad, lens, Y, tgt_lens
