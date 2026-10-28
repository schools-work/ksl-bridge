"""Okapi BM25 (no neural net; strong on keyword-heavy menus). In-memory, for small corpora and tests."""

from __future__ import annotations

import math
import re
from collections import Counter

TOK = re.compile(r"[a-z0-9]+")


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [TOK.findall(d.lower()) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / max(1, self.N)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b = k1, b

    def idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, q: str) -> list[float]:
        qt = TOK.findall(q.lower())
        out = []
        for d, tf in zip(self.docs, self.tf, strict=True):
            s = 0.0
            for t in qt:
                f = tf.get(t, 0)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out

    def top(self, q: str, k: int = 5) -> list[tuple[int, float]]:
        sc = self.scores(q)
        return sorted(((i, s) for i, s in enumerate(sc) if s > 0), key=lambda x: -x[1])[:k]
