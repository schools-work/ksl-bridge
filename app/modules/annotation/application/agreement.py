"""Inter-annotator agreement: Cohen's kappa (categorical) and gloss edit agreement (sequences)."""

from __future__ import annotations

from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _edit(r: list[str], h: list[str]) -> int:
    prev = list(range(len(h) + 1))
    for i, x in enumerate(r, 1):
        cur = [i]
        for j, y in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def gloss_edit_agreement(pairs: list[tuple[list[str], list[str]]]) -> float:
    """1 − (total edits / total reference length); symmetric by averaging both directions."""
    edits = sum(_edit(a, b) for a, b in pairs)
    length = sum((len(a) + len(b)) / 2 for a, b in pairs)
    return 1.0 - edits / max(length, 1)
