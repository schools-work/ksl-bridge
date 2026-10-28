"""Isolated clips → pseudo-sentences with cross-faded transitions (labelled is_synthetic; never evaluates)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def crossfade_concat(clips: list[np.ndarray], fade: int = 6) -> np.ndarray:
    out = clips[0]
    for nxt in clips[1:]:
        k = min(fade, len(out), len(nxt))
        w = np.linspace(0, 1, k)[:, None, None]
        blend = (1 - w) * out[-k:] + w * nxt[:k]
        out = np.concatenate([out[:-k], blend, nxt[k:]], axis=0)
    return out.astype(np.float32)


def synthesize(
    sample_sentence: Callable[[], list[str]],
    clips_by_gloss: dict[str, list[np.ndarray]],
    n: int,
    rng: np.random.Generator | None = None,
) -> list[dict[str, object]]:
    rng = rng or np.random.default_rng(0)
    out: list[dict[str, object]] = []
    while len(out) < n:
        glosses = [g for g in sample_sentence() if g in clips_by_gloss]
        if len(glosses) < 2:
            continue
        chosen = [clips_by_gloss[g][int(rng.integers(len(clips_by_gloss[g])))] for g in glosses]
        out.append({"xyz": crossfade_concat(chosen, int(rng.integers(4, 9))), "glosses": glosses, "is_synthetic": True})
    return out
