"""Fit temperature (token level) and the utterance-level logistic calibrator on validation outputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ml.decoding.calibration import UTTERANCE_FEATURES, expected_calibration_error, fit_utterance_calibrator


def main() -> None:
    ap = argparse.ArgumentParser(description="input: JSONL with feature columns + 'correct' (0/1)")
    ap.add_argument("val_jsonl")
    ap.add_argument("--out", default="artifacts/calibration/utterance_lr.pkl")
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.val_jsonl).read_text().splitlines() if line.strip()]
    X = np.array([[r[f] for f in UTTERANCE_FEATURES] for r in rows], dtype=np.float32)
    y = np.array([int(r["correct"]) for r in rows])
    raw = 1 / (1 + np.exp(-X[:, 0]))
    clf = fit_utterance_calibrator(X, y)
    cal = clf.predict_proba(X)[:, 1]
    print(f"ECE raw {expected_calibration_error(raw, y):.3f} → calibrated {expected_calibration_error(cal, y):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(clf, f)


if __name__ == "__main__":
    main()
