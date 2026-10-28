"""CTC prefix beam search with LM shallow fusion and context bias (informed search; Guide Part V §3.2)."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NEG_INF = -float("inf")


def lse(*xs: float) -> float:
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass
class Beam:
    pb: float = NEG_INF
    pnb: float = NEG_INF

    def acoustic(self) -> float:
        return lse(self.pb, self.pnb)


@dataclass
class CTCPrefixBeamSearch:
    itos: list[str]
    lm: Any | None = None
    beam: int = 8
    alpha: float = 0.6
    beta: float = 1.0
    prune_logp: float = -8.0
    blank: int = 0
    skip_ids: frozenset[int] = frozenset({1, 3, 4})  # <pad>,<bos>,<eos> never emitted; <unk> kept on purpose
    context_bias: dict[str, float] = field(default_factory=dict)
    beams: dict[tuple[int, ...], Beam] = field(default_factory=lambda: {(): Beam(pb=0.0)})
    lm_cache: dict[tuple[int, ...], float] = field(default_factory=lambda: {(): 0.0})

    def _extend_lm(self, prefix: tuple[int, ...], new: tuple[int, ...]) -> None:
        if new in self.lm_cache:
            return
        g = self.itos[new[-1]]
        s = self.beta + self.context_bias.get(g, 0.0)
        if self.lm is not None:
            s += self.alpha * self.lm.logp([self.itos[i] for i in prefix], g)
        self.lm_cache[new] = self.lm_cache[prefix] + s

    def _total(self, prefix: tuple[int, ...], b: Beam) -> float:
        return b.acoustic() + self.lm_cache[prefix]

    def step(self, log_probs: np.ndarray) -> None:
        for t in range(log_probs.shape[0]):
            lp = log_probs[t]
            cand = np.where(lp > self.prune_logp)[0]
            if len(cand) == 0:
                cand = np.argsort(lp)[-self.beam :]
            cands = [int(c) for c in cand if int(c) not in self.skip_ids]
            nxt: dict[tuple[int, ...], Beam] = defaultdict(Beam)
            for prefix, b in self.beams.items():
                for c in cands:
                    p = float(lp[c])
                    if c == self.blank:
                        nb = nxt[prefix]
                        nb.pb = lse(nb.pb, b.pb + p, b.pnb + p)
                        continue
                    last = prefix[-1] if prefix else None
                    new = (*prefix, c)
                    self._extend_lm(prefix, new)
                    nn_ = nxt[new]
                    if c == last:
                        nn_.pnb = lse(nn_.pnb, b.pb + p)
                        same = nxt[prefix]
                        same.pnb = lse(same.pnb, b.pnb + p)
                    else:
                        nn_.pnb = lse(nn_.pnb, b.pb + p, b.pnb + p)
            self.beams = dict(sorted(nxt.items(), key=lambda kv: -self._total(*kv))[: self.beam])

    def nbest(self, n: int = 5, final: bool = False) -> list[tuple[list[str], float]]:
        scored = []
        for prefix, b in self.beams.items():
            s = self._total(prefix, b)
            if final and self.lm is not None:
                s += self.alpha * self.lm.logp([self.itos[i] for i in prefix], "</s>")
            scored.append(([self.itos[i] for i in prefix], s))
        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    def reset(self) -> None:
        self.beams = {(): Beam(pb=0.0)}
        self.lm_cache = {(): 0.0}


def greedy_decode(log_probs: np.ndarray, itos: list[str], blank: int = 0) -> list[str]:
    out, prev = [], None
    for t in log_probs.argmax(-1):
        if t != prev and t != blank:
            out.append(itos[int(t)])
        prev = t
    return out
