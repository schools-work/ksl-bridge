"""Joint final decision over N-best × translation × NLU (Guide Part V §3.4). Pure."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    glosses: tuple[str, ...]
    cslr_score: float
    text: str
    mt_logp: float
    intent: str
    intent_conf: float


def choose_final(cands: list[Candidate], w: tuple[float, float, float] = (1.0, 0.5, 0.8)) -> Candidate:
    if not cands:
        raise ValueError("no candidates")

    def score(c: Candidate) -> float:
        return w[0] * c.cslr_score + w[1] * c.mt_logp + w[2] * math.log(max(c.intent_conf, 1e-6))

    return max(cands[:3], key=score)


def softmax_confidence(scores: list[float]) -> list[float]:
    """Turn N-best log scores into a distribution (pre-calibration)."""
    if not scores:
        return []
    m = max(scores)
    ex = [math.exp(s - m) for s in scores]
    z = sum(ex)
    return [e / z for e in ex]
