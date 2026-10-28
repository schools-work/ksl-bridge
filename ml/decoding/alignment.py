"""Viterbi forced alignment of the best hypothesis → per-token spans, confidence and N-best alternatives."""

from __future__ import annotations

from typing import Any

import numpy as np


def ctc_force_align(log_probs: np.ndarray, target: list[int], blank: int = 0) -> list[tuple[int, int]]:
    """Return (start_frame, end_frame) per target token via CTC Viterbi over the extended label sequence."""
    T = log_probs.shape[0]
    ext = [blank]
    for t in target:
        ext += [t, blank]
    S = len(ext)
    if T == 0 or not target:
        return []
    dp = np.full((T, S), -np.inf)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        dp[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp[t - 1, s], s)]
            if s >= 1:
                cands.append((dp[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                cands.append((dp[t - 1, s - 2], s - 2))
            v, arg = max(cands)
            dp[t, s] = v + log_probs[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if S < 2 or dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s = int(bp[t, s])
        path.append(s)
    path.reverse()
    spans: dict[int, list[int]] = {}
    for t, s in enumerate(path):
        if ext[s] != blank:
            spans.setdefault(s // 2, []).append(t)
    return [(min(v), max(v)) for _, v in sorted(spans.items())]


def _levenshtein_align(a: list[str], b: list[str]) -> dict[int, str]:
    """Map positions of a → aligned token of b (substitutions/matches only)."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out[i - 1] = b[j - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def token_spans(
    log_probs: np.ndarray, itos: list[str], nbest: list[tuple[list[str], float]], frame_ms: float
) -> list[dict[str, Any]]:
    if not nbest or not nbest[0][0]:
        return []
    stoi = {s: i for i, s in enumerate(itos)}
    best = nbest[0][0]
    target = [stoi[g] for g in best if g in stoi]
    spans = ctc_force_align(log_probs, target)
    probs = np.exp(log_probs)
    scores = np.array([s for _, s in nbest])
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    alts: list[dict[str, float]] = [{} for _ in best]
    for (hyp, _), w in zip(nbest[1:], weights[1:], strict=True):
        for pos, tok in _levenshtein_align(best, hyp).items():
            if tok != best[pos]:
                alts[pos][tok] = alts[pos].get(tok, 0.0) + float(w)
    out = []
    for k, (tok, (s, e)) in enumerate(zip(best, spans, strict=False)):
        conf = float(probs[s : e + 1, stoi[tok]].mean()) if tok in stoi else 0.0
        out.append(
            {
                "token": tok,
                "start_ms": int(s * frame_ms),
                "end_ms": int((e + 1) * frame_ms),
                "confidence": max(0.0, min(1.0, conf)),
                "alternatives": sorted(alts[k].items(), key=lambda kv: -kv[1])[:3],
            }
        )
    return out
