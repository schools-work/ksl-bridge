"""Landmark smoothing filters (AIMA §15.4 Kalman; One-Euro for low-lag jitter removal)."""

from __future__ import annotations

import math

import numpy as np


class ConstantVelocityKalman:
    """Independent 1-D constant-velocity Kalman filters per coordinate. State = [pos, vel]."""

    def __init__(self, n: int, q: float = 1e-3, r: float = 4e-3, dt: float = 1 / 30):
        self.F = np.array([[1, dt], [0, 1]], dtype=np.float32)
        self.Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float32)
        self.R = r
        self.x = np.zeros((n, 2), np.float32)
        self.P = np.tile(np.eye(2, dtype=np.float32), (n, 1, 1))
        self.init = np.zeros(n, bool)

    def step(self, z: np.ndarray) -> np.ndarray:
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        obs = np.isfinite(z)
        first = obs & ~self.init
        self.x[first, 0], self.x[first, 1] = z[first], 0.0
        self.init |= obs
        upd = obs & ~first
        if upd.any():
            P = self.P[upd]
            S = P[:, 0, 0] + self.R
            K = P[:, :, 0] / S[:, None]
            y = z[upd] - self.x[upd, 0]
            self.x[upd] += K * y[:, None]
            self.P[upd] = P - K[:, :, None] * P[:, 0:1, :]
        out = self.x[:, 0].copy()
        out[~self.init] = np.nan
        return out


class OneEuroFilter:
    """One-Euro filter (Casiez et al., 2012), vectorised over coordinates."""

    def __init__(self, n: int, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.01, d_cutoff: float = 1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = np.full(n, np.nan, np.float32)
        self.dx_prev = np.zeros(n, np.float32)

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, freq: float) -> np.ndarray:
        tau = 1.0 / (2 * math.pi * np.asarray(cutoff))
        return 1.0 / (1.0 + tau * freq)

    def step(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        fresh = ~np.isfinite(self.x_prev) & np.isfinite(x)
        self.x_prev[fresh] = x[fresh]
        dx = np.where(np.isfinite(x), (x - self.x_prev) * self.freq, 0.0)
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = np.where(np.isfinite(x), a * x + (1 - a) * self.x_prev, self.x_prev)
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat
