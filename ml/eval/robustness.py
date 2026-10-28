"""Robustness matrix (RQ2): controlled perturbations of the test set (Guide Part VIII §2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml.data import augment as A

from .metrics import bootstrap_ci, wer

PERTURBATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "clean": lambda x: x,
    "rot15": lambda x: A.rotate3d(x, 15),
    "rot30": lambda x: A.rotate3d(x, 30),
    "slow": lambda x: A.time_warp(x, 1.3, 1.3),
    "fast": lambda x: A.time_warp(x, 0.7, 0.7),
    "jitter02": lambda x: A.finger_jitter(x, 0.02),
    "jitter05": lambda x: A.finger_jitter(x, 0.05),
    "drop10": lambda x: A.hand_dropout(x, p_frame=0.1),
    "drop30": lambda x: A.hand_dropout(x, p_frame=0.3),
}


def robustness_report(
    model_fn: Callable[[np.ndarray], list[str]], test_items: list[dict[str, Any]], seed: int = 0
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, f in PERTURBATIONS.items():
        A.seed(seed)
        pairs = [(it["glosses"], model_fn(f(np.asarray(it["xyz"], dtype=np.float32)))) for it in test_items]
        res = wer([p[0] for p in pairs], [p[1] for p in pairs])
        lo, hi = bootstrap_ci(pairs, lambda ps: wer([p[0] for p in ps], [p[1] for p in ps])["wer"], n=200)
        rows[name] = {**res, "ci_low": lo, "ci_high": hi, "n": len(pairs)}
    clean = rows["clean"]["wer"]
    for r in rows.values():
        r["delta_vs_clean"] = r["wer"] - clean
    return rows
