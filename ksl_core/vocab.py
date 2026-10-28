"""Token vocabularies. <blank> must be index 0 (CTC). Hash must match ml.vocabularies.sha256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SPECIALS = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>"]
BLANK, PAD, UNK, BOS, EOS = range(5)


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.itos[: len(SPECIALS)] != SPECIALS:
            raise ValueError("vocabulary must start with the special tokens in order")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("duplicate tokens in vocabulary")
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, tokens: list[str]) -> Vocab:
        uniq = sorted(set(tokens) - set(SPECIALS))
        return cls(SPECIALS + uniq)

    def encode(self, seq: list[str]) -> list[int]:
        return [self.stoi.get(g, UNK) for g in seq]

    def decode(self, ids: list[int], keep_unk: bool = False) -> list[str]:
        keep = {UNK} if keep_unk else set()
        return [self.itos[i] for i in ids if i >= len(SPECIALS) or i in keep]

    def sha256(self) -> bytes:
        return hashlib.sha256("\n".join(self.itos).encode()).digest()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        return cls(json.loads(Path(path).read_text()))
