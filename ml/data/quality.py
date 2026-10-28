"""Per-sample quality checks → dataset.samples.quality_flags."""

from __future__ import annotations

import numpy as np

from ksl_core.landmarks import SL_LH, SL_RH


def quality_flags(xyz: np.ndarray, fps: float, min_visible: float = 0.8) -> list[str]:
    flags: list[str] = []
    if xyz.shape[0] < 8:
        flags.append("too_short")
    if fps < 20:
        flags.append("low_fps")
    hands = np.concatenate([xyz[:, SL_LH], xyz[:, SL_RH]], axis=1)
    any_hand = np.isfinite(hands).all(-1).any(-1)
    if any_hand.mean() < min_visible:
        flags.append("hand_out_of_frame")
    if np.isfinite(xyz).sum() and np.nanstd(np.diff(xyz, axis=0)) > 0.2:
        flags.append("jittery")
    return flags
