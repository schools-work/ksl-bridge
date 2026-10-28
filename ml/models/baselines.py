"""Course baselines: k-NN + DTW and Random Forest on handshape statistics (Guide Part IV §1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    Ta, Tb = len(a), len(b)
    w = max(window or max(Ta, Tb), abs(Ta - Tb))
    D = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        lo, hi = max(1, i - w), min(Tb, i + w)
        cost = np.linalg.norm(b[lo - 1 : hi] - a[i - 1], axis=1)
        for k, j in enumerate(range(lo, hi + 1)):
            D[i, j] = cost[k] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[Ta, Tb] / (Ta + Tb))


class KnnDTW:
    def __init__(self, k: int = 3, window: int = 20, pca_dim: int | None = 64):
        self.k, self.window, self.pca_dim = k, window, pca_dim
        self.pca: Any = None

    def _proj(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(x) if self.pca is not None else x

    def fit(self, X: list[np.ndarray], y: list[str]) -> KnnDTW:
        if self.pca_dim:
            from sklearn.decomposition import PCA

            stacked = np.concatenate(X)
            self.pca = PCA(n_components=min(self.pca_dim, stacked.shape[1], len(stacked))).fit(stacked)
        self.X, self.y = [self._proj(x) for x in X], y
        return self

    def predict_proba(self, q: np.ndarray) -> dict[str, float]:
        q = self._proj(q)
        d = np.array([dtw_distance(q, x, self.window) for x in self.X])
        idx = np.argsort(d)[: self.k]
        w = np.exp(-d[idx] / (d[idx].mean() + 1e-6))
        out: dict[str, float] = {}
        for i, wi in zip(idx, w, strict=True):
            out[self.y[i]] = out.get(self.y[i], 0.0) + float(wi)
        s = sum(out.values())
        return {k: v / s for k, v in sorted(out.items(), key=lambda kv: -kv[1])}

    def predict(self, q: np.ndarray) -> str:
        return next(iter(self.predict_proba(q)))


def summary_features(feats: np.ndarray) -> np.ndarray:
    """Clip-level statistics of hand-local shape + bone features (columns 642:888) + motion energy."""
    hs = feats[:, 642:888]
    energy = np.linalg.norm(feats[:, 321:642], axis=1)
    return np.concatenate([hs.mean(0), hs.std(0), hs.min(0), hs.max(0), [energy.mean(), energy.std(), len(feats)]])


def random_forest(n_estimators: int = 500, seed: int = 0) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed, class_weight="balanced")
