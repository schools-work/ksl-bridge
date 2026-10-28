"""Parallel-data utilities for gloss⇄text training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_pairs(path: str | Path) -> list[tuple[list[str], str]]:
    """TSV: glosses<TAB>english"""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            g, t = line.split("\t", 1)
            out.append((g.split(), t.strip()))
    return out


class PairDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, pairs: list[tuple[list[str], str]], src_stoi: dict[str, int], sp: Any, reverse: bool = False):
        self.pairs, self.src_stoi, self.sp, self.reverse = pairs, src_stoi, sp, reverse

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        g, t = self.pairs[i]
        gl = [self.src_stoi.get(x, 2) for x in g]
        tx = self.sp.encode(t)
        src, tgt = (tx, gl) if self.reverse else (gl, tx)
        return torch.tensor(src), torch.tensor([3, *tgt, 4])


def collate(b: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    s, t = zip(*b, strict=True)
    pad = torch.nn.utils.rnn.pad_sequence
    return pad(list(s), batch_first=True, padding_value=1), pad(list(t), batch_first=True, padding_value=1)
