"""Reliability diagram data + ECE before/after calibration."""

from __future__ import annotations

import numpy as np

from ml.decoding.calibration import expected_calibration_error


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append(
                {
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "confidence": float(conf[m].mean()),
                    "accuracy": float(correct[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return rows


def report(raw: np.ndarray, calibrated: np.ndarray, correct: np.ndarray) -> dict[str, object]:
    return {
        "ece_before": expected_calibration_error(raw, correct),
        "ece_after": expected_calibration_error(calibrated, correct),
        "reliability_after": reliability(calibrated, correct),
    }
