"""Signer-independent splits (dataset.enforce_build_member is the database's second line of defence)."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


def signer_split(
    samples: list[dict[str, Any]], test_frac: float = 0.15, val_frac: float = 0.10, seed: str = "ksl-v1"
) -> dict[str, list[dict[str, Any]]]:
    by_signer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_signer[s["signer_id"]].append(s)

    def h(sid: str) -> float:
        return int(hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for sid, items in by_signer.items():
        r = h(sid)
        split = "test" if r < test_frac else "val" if r < test_frac + val_frac else "train"
        out[split].extend(items)
    return out


def coverage_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Every test gloss must appear in train; report group balance."""
    train_g = {g for s in splits["train"] for g in s["glosses"]}
    test_g = {g for s in splits["test"] for g in s["glosses"]}
    hands = {k: sorted({s.get("dominant_hand") for s in v}, key=str) for k, v in splits.items()}
    return {
        "test_glosses_missing_in_train": sorted(test_g - train_g),
        "dominant_hands_per_split": hands,
        "signers": {k: len({s["signer_id"] for s in v}) for k, v in splits.items()},
    }
