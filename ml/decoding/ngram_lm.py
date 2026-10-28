"""Interpolated Kneser–Ney n-gram LM (AIMA §22.1; Chen & Goodman 1999) with domain interpolation."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS = "<s>", "</s>"


class KneserNeyLM:
    def __init__(self, order: int = 4, discount: float = 0.75):
        self.n, self.D = order, discount
        self.tab: list[dict[tuple[str, ...], Counter[str]]] = []
        self.tot: list[dict[tuple[str, ...], int]] = []
        self.V = 1
        self._cache: dict[tuple[int, tuple[str, ...], str], float] = {}

    def fit(self, sentences: list[list[str]]) -> KneserNeyLM:
        n = self.n
        raw: list[defaultdict[tuple[str, ...], Counter[str]]] = [defaultdict(Counter) for _ in range(n + 1)]
        vocab: set[str] = set()
        for s in sentences:
            toks = [BOS] * (n - 1) + s + [EOS]
            vocab.update(s)
            vocab.add(EOS)
            for i in range(n - 1, len(toks)):
                for k in range(1, n + 1):
                    raw[k][tuple(toks[i - k + 1 : i])][toks[i]] += 1
        self.tab = [{} for _ in range(n + 1)]
        self.tab[n] = dict(raw[n])
        for k in range(n - 1, 0, -1):
            cont: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
            for ctx, words in raw[k + 1].items():
                for w in words:
                    cont[ctx[1:]][w] += 1
            self.tab[k] = dict(cont)
        self.tot = [{}] + [{c: sum(ws.values()) for c, ws in self.tab[k].items()} for k in range(1, n + 1)]
        self.V = len(vocab) + 1
        self._cache.clear()
        return self

    def _p(self, k: int, ctx: tuple[str, ...], w: str) -> float:
        if k == 0:
            return 1.0 / self.V
        key = (k, ctx, w)
        if key in self._cache:
            return self._cache[key]
        lower = self._p(k - 1, ctx[1:], w)
        table = self.tab[k].get(ctx)
        if not table:
            p = lower
        else:
            total, types = self.tot[k][ctx], len(table)
            p = max(table.get(w, 0) - self.D, 0) / total + self.D * types / total * lower
        self._cache[key] = p
        return p

    def logp(self, history: list[str] | tuple[str, ...], w: str) -> float:
        ctx = tuple(([BOS] * (self.n - 1) + list(history))[-(self.n - 1) :]) if self.n > 1 else ()
        return math.log(self._p(self.n, ctx, w))

    def perplexity(self, sentences: list[list[str]]) -> float:
        lp, N = 0.0, 0
        for s in sentences:
            for i, w in enumerate([*s, EOS]):
                lp += self.logp(s[:i], w)
                N += 1
        return math.exp(-lp / max(N, 1))

    def save(self, path: str | Path) -> None:
        data = {
            "n": self.n,
            "D": self.D,
            "V": self.V,
            "tab": [[[list(c), dict(ws)] for c, ws in t.items()] for t in self.tab],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> KneserNeyLM:
        d = json.loads(Path(path).read_text())
        lm = cls(d["n"], d["D"])
        lm.V = d["V"]
        lm.tab = [{tuple(c): Counter(ws) for c, ws in t} for t in d["tab"]]
        lm.tot = [{}] + [{c: sum(ws.values()) for c, ws in lm.tab[k].items()} for k in range(1, lm.n + 1)]
        return lm


class InterpolatedLM:
    """P = λ·P_domain + (1−λ)·P_general (rooms select the domain by venue type)."""

    def __init__(self, domain: KneserNeyLM, general: KneserNeyLM, lam: float = 0.6):
        self.d, self.g, self.lam = domain, general, lam

    def logp(self, history: list[str] | tuple[str, ...], w: str) -> float:
        return math.log(
            self.lam * math.exp(self.d.logp(history, w)) + (1 - self.lam) * math.exp(self.g.logp(history, w))
        )
