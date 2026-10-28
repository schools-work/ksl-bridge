"""WER/CER, accuracy, BLEU/chrF (sacrebleu), ECE, bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int]:
    R, H = len(ref), len(hyp)
    D = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        D[i][0] = i
    for j in range(H + 1):
        D[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, Dl, I = R, H, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            Dl, i = Dl + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, Dl, I


def wer(refs: list[list[str]], hyps: list[list[str]]) -> dict[str, float]:
    S = Dl = I = N = 0
    for r, h in zip(refs, hyps, strict=True):
        s, d, i = edit_ops(r, h)
        S, Dl, I, N = S + s, Dl + d, I + i, N + len(r)
    N = max(N, 1)
    return {"wer": (S + Dl + I) / N, "sub": S / N, "del": Dl / N, "ins": I / N}


def cer(refs: list[str], hyps: list[str]) -> float:
    return wer([list(r) for r in refs], [list(h) for h in hyps])["wer"]


def accuracy(y: Sequence[Any], yhat: Sequence[Any]) -> float:
    return float(np.mean([a == b for a, b in zip(y, yhat, strict=True)])) if y else 0.0


def bleu_chrf(refs: list[str], hyps: list[str]) -> dict[str, float]:
    import sacrebleu

    return {
        "bleu": float(sacrebleu.corpus_bleu(hyps, [refs]).score),
        "chrf": float(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score),
    }


def bootstrap_ci(
    values: Sequence[Any], stat: Callable[[list[Any]], float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = list(values)
    stats = sorted(stat([vals[i] for i in rng.integers(0, len(vals), len(vals))]) for _ in range(n))
    return stats[int(n * alpha / 2)], stats[int(n * (1 - alpha / 2)) - 1]
