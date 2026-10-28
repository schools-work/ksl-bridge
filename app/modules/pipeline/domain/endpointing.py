"""End-of-utterance detection from hand motion energy and rest position (Guide Part V §3.1)."""

from __future__ import annotations

import numpy as np

from ksl_core.landmarks import SL_LH, SL_RH


class Endpointer:
    def __init__(self, fps: float = 30, still_ms: int = 450, energy_thr: float = 0.015, rest_y: float = 1.6):
        self.need = max(1, int(still_ms / 1000 * fps))
        self.thr, self.rest_y = energy_thr, rest_y
        self.still, self.active = 0, False
        self.prev: np.ndarray | None = None

    def update(self, frame_norm: np.ndarray) -> str | None:
        """frame_norm: (107,3) normalised. Returns 'start', 'end' or None."""
        hands = np.concatenate([frame_norm[SL_LH], frame_norm[SL_RH]])
        visible = np.isfinite(hands).all(-1)
        at_rest = (not visible.any()) or float(np.nanmean(hands[visible][:, 1])) > self.rest_y
        energy = 0.0
        if self.prev is not None and visible.any():
            d = hands - self.prev
            with np.errstate(all="ignore"):
                e = np.nanmean(np.linalg.norm(d, axis=-1))
            energy = float(e) if np.isfinite(e) else 0.0
        self.prev = hands
        moving = energy > self.thr and not at_rest
        if not self.active and moving:
            self.active, self.still = True, 0
            return "start"
        if self.active:
            self.still = 0 if moving else self.still + 1
            if self.still >= self.need:
                self.active = False
                return "end"
        return None
