"""Confidence calibration: temperature scaling, ECE, utterance-level logistic regression (Guide Part V §4.1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def fit_temperature(logits: Any, labels: Any, iters: int = 200) -> float:
    import torch

    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)
    nll = torch.nn.CrossEntropyLoss()

    def closure() -> Any:
        opt.zero_grad()
        loss = nll(logits / logT.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.detach().exp())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


UTTERANCE_FEATURES = ("norm_score", "margin", "lm_logp", "min_token_post", "n_unk_fs", "mt_logp", "hand_visibility")


def utterance_features(
    nbest: list[tuple[list[str], float]],
    spans: list[dict[str, Any]],
    lm_logp: float,
    mt_logp: float,
    hand_visibility: float,
) -> np.ndarray:
    best_len = max(1, len(nbest[0][0])) if nbest else 1
    s0 = nbest[0][1] if nbest else -50.0
    s1 = nbest[1][1] if len(nbest) > 1 else s0 - 10.0
    min_post = min((sp["confidence"] for sp in spans), default=0.0)
    n_unk = sum(1 for g in (nbest[0][0] if nbest else []) if g == "<unk>" or g.startswith("FS:"))
    return np.array([s0 / best_len, s0 - s1, lm_logp, min_post, n_unk, mt_logp, hand_visibility], dtype=np.float32)


def fit_utterance_calibrator(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
