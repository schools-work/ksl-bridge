"""Augmentations = the 'small mistakes' simulator (Guide Part III §5). Applied to raw landmarks."""

from __future__ import annotations

import numpy as np

from ksl_core.landmarks import SL_LH, SL_RH

rng = np.random.default_rng()


def seed(s: int) -> None:
    global rng
    rng = np.random.default_rng(s)


def rotate3d(x: np.ndarray, max_deg: float = 15) -> np.ndarray:
    a, b, c = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return (x @ (Rz @ Ry @ Rx).T).astype(np.float32)


def scale_shift(x: np.ndarray, s: float = 0.1, t: float = 0.1) -> np.ndarray:
    return (x * rng.uniform(1 - s, 1 + s) + rng.uniform(-t, t, size=(1, 1, 3))).astype(np.float32)


def time_warp(x: np.ndarray, lo: float = 0.7, hi: float = 1.3) -> np.ndarray:
    T = x.shape[0]
    newT = max(4, int(round(T * rng.uniform(lo, hi))))
    src = np.linspace(0, T - 1, newT)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None, None]
    return ((1 - w) * x[i0] + w * x[i1]).astype(np.float32)


def finger_jitter(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    y = x.copy()
    for sl in (SL_LH, SL_RH):
        y[:, sl] += rng.normal(0, sigma, size=y[:, sl].shape).astype(np.float32)
    return y


def hand_dropout(x: np.ndarray, p_frame: float = 0.1, max_span: int = 8) -> np.ndarray:
    y = x.copy()
    T = x.shape[0]
    for sl in (SL_LH, SL_RH):
        if rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 3))):
                L = int(rng.integers(1, max_span + 1))
                s = int(rng.integers(0, max(1, T - L)))
                y[s : s + L, sl] = np.nan
        y[rng.random(T) < p_frame, sl] = np.nan
    return y


def mirror(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., 0] *= -1
    lh = y[:, SL_LH].copy()
    y[:, SL_LH] = y[:, SL_RH]
    y[:, SL_RH] = lh
    return y


def temporal_crop(x: np.ndarray, min_keep: float = 0.85) -> np.ndarray:
    T = x.shape[0]
    keep = max(4, int(T * rng.uniform(min_keep, 1.0)))
    s = int(rng.integers(0, T - keep + 1))
    return x[s : s + keep]


def augment(x: np.ndarray, allow_mirror: bool) -> np.ndarray:
    """allow_mirror must be False when any gloss is lexicon.glosses.is_directional."""
    if rng.random() < 0.8:
        x = rotate3d(x)
    if rng.random() < 0.8:
        x = scale_shift(x)
    if rng.random() < 0.6:
        x = time_warp(x)
    if rng.random() < 0.5:
        x = finger_jitter(x)
    if rng.random() < 0.5:
        x = hand_dropout(x)
    if rng.random() < 0.3:
        x = temporal_crop(x)
    if allow_mirror and rng.random() < 0.5:
        x = mirror(x)
    return x
