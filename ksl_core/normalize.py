"""Body-centred pose normalisation and feature building (Guide Part III §3).

Used identically by training (ml/data) and serving (app pipeline + workers): the main defence
against training/serving skew.
"""

from __future__ import annotations

import numpy as np

from .landmarks import POSE_L_SHOULDER, POSE_NOSE, POSE_R_SHOULDER, SL_FACE, SL_LH, SL_POSE, SL_RH

EPS = 1e-6
FEATURE_DIM = 892

HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def interp_gaps(x: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """Linearly interpolate interior NaN runs of length ≤ max_gap along time. x: (T, P, 3)."""
    T = x.shape[0]
    out = x.copy()
    flat = out.reshape(T, -1)
    nan = np.isnan(flat)
    if not nan.any():
        return out
    t = np.arange(T)
    for j in np.where(nan.any(0) & ~nan.all(0))[0]:
        col, miss = flat[:, j], nan[:, j]
        good = ~miss
        filled = np.interp(t, t[good], col[good])
        # run-length encode missing runs; fill only bounded runs ≤ max_gap
        d = np.diff(np.concatenate([[0], miss.astype(np.int8), [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for s, e in zip(starts, ends, strict=True):
            if s > 0 and e < T and (e - s) <= max_gap:
                col[s:e] = filled[s:e]
    return out


def _mirror(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., 0] *= -1.0
    lh = y[:, SL_LH].copy()
    y[:, SL_LH] = y[:, SL_RH]
    y[:, SL_RH] = lh
    return y


def normalize_sequence(xyz: np.ndarray, mirror: bool = False, max_gap: int = 6) -> np.ndarray:
    """(T,107,3) raw image coords (NaN = missing) → body-centred, shoulder-scaled, roll-corrected."""
    x = interp_gaps(xyz.astype(np.float32), max_gap)
    pose = x[:, SL_POSE]
    ls, rs = pose[:, POSE_L_SHOULDER], pose[:, POSE_R_SHOULDER]
    with np.errstate(all="ignore"):
        center = np.nanmedian((ls + rs) / 2.0, axis=0)
        width = np.nanmedian(np.linalg.norm((ls - rs)[:, :2], axis=-1))
        v = np.nanmedian((ls - rs)[:, :2], axis=0)
    if not np.isfinite(center).all():
        center = np.zeros(3, np.float32)
    if not np.isfinite(width) or width < EPS:
        width = 0.25
    x = (x - center) / width
    if np.isfinite(v).all():
        angle = np.arctan2(v[1], v[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        x[..., :2] = x[..., :2] @ R.T
    return _mirror(x) if mirror else x


class StreamingNormalizer:
    """Per-session online normaliser: EMA of shoulder centre/width/roll (Guide Part III §3.3)."""

    def __init__(self, alpha: float = 0.05, warmup: int = 15):
        self.alpha, self.warmup = alpha, warmup
        self.n = 0
        self.center = np.zeros(3, np.float32)
        self.width = 0.25
        self.vec = np.array([1.0, 0.0], np.float32)

    def update(self, frames: np.ndarray, mirror: bool = False) -> np.ndarray:
        out = np.empty_like(frames, dtype=np.float32)
        for i, f in enumerate(frames):
            ls, rs = f[SL_POSE][POSE_L_SHOULDER], f[SL_POSE][POSE_R_SHOULDER]
            if np.isfinite(ls).all() and np.isfinite(rs).all():
                a = 1.0 / (self.n + 1) if self.n < self.warmup else self.alpha
                self.center = (1 - a) * self.center + a * (ls + rs) / 2
                w = float(np.linalg.norm((ls - rs)[:2]))
                if w > EPS:
                    self.width = (1 - a) * self.width + a * w
                self.vec = (1 - a) * self.vec + a * (ls - rs)[:2]
                self.n += 1
            y = (f - self.center) / self.width
            ang = np.arctan2(self.vec[1], self.vec[0])
            c, s = np.cos(-ang), np.sin(-ang)
            y[:, :2] = y[:, :2] @ np.array([[c, -s], [s, c]], np.float32).T
            out[i] = y
        return _mirror(out) if mirror else out


def hand_local(hand: np.ndarray) -> np.ndarray:
    """(T,21,3) → wrist-centred, palm-scaled (handshape only)."""
    wrist = hand[:, 0:1]
    palm = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=-1)[:, None, None]
    out: np.ndarray = (hand - wrist) / np.maximum(np.nan_to_num(palm, nan=1.0), EPS)
    return out


def build_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised (T,107,3) → features (T, 892) float32 and presence mask (T, 892) bool."""
    T = x.shape[0]
    lh, rh, face = x[:, SL_LH], x[:, SL_RH], x[:, SL_FACE]
    pos = x.reshape(T, -1)
    vel = np.diff(x, axis=0, prepend=x[:1]).reshape(T, -1)
    shape = np.concatenate([hand_local(lh).reshape(T, -1), hand_local(rh).reshape(T, -1)], -1)
    bones = []
    for hand in (lh, rh):
        for a, b in HAND_BONES:
            d = hand[:, b] - hand[:, a]
            bones.append(d / np.maximum(np.nan_to_num(np.linalg.norm(d, axis=-1, keepdims=True), nan=1.0), EPS))
    bones_arr = np.concatenate(bones, axis=-1)
    nose = x[:, SL_POSE][:, POSE_NOSE]
    dists = np.stack(
        [
            np.linalg.norm(lh[:, 0] - rh[:, 0], axis=-1),
            np.linalg.norm(lh[:, 8] - nose, axis=-1),
            np.linalg.norm(rh[:, 8] - nose, axis=-1),
            np.linalg.norm(face[:, 21] - face[:, 26], axis=-1),
        ],
        axis=-1,
    )
    feats = np.concatenate([pos, vel, shape, bones_arr, dists], axis=-1).astype(np.float32)
    assert feats.shape[1] == FEATURE_DIM, feats.shape
    mask = np.isfinite(feats)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), mask
