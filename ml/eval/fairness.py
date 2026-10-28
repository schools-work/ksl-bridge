"""Per-group slices and worst-case gaps (Brookings-style audit)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .metrics import wer

GROUP_KEYS = (
    "gender",
    "age_band",
    "region",
    "dominant_hand",
    "hearing_status",
    "device_tier",
    "lighting",
    "skin_tone_band",
)


def group_wer(items: list[dict[str, Any]], hyps: list[list[str]], group_key: str) -> tuple[dict[str, float], float]:
    by: dict[str, tuple[list[list[str]], list[list[str]]]] = defaultdict(lambda: ([], []))
    for it, h in zip(items, hyps, strict=True):
        g = str(it.get("meta", {}).get(group_key, "unknown"))
        by[g][0].append(it["glosses"])
        by[g][1].append(h)
    res = {g: wer(r, h)["wer"] for g, (r, h) in by.items()}
    return res, (max(res.values()) - min(res.values())) if res else 0.0


def fairness_rows(items: list[dict[str, Any]], hyps: list[list[str]]) -> list[dict[str, Any]]:
    """Rows ready for ml.evaluation_results (slice_key/slice_value)."""
    rows: list[dict[str, Any]] = []
    for key in GROUP_KEYS:
        if not any(key in it.get("meta", {}) for it in items):
            continue
        res, _ = group_wer(items, hyps, key)
        for g, v in res.items():
            n = sum(1 for it in items if str(it.get("meta", {}).get(key, "unknown")) == g)
            rows.append({"metric": "wer", "slice_key": key, "slice_value": g, "value": v, "n": n})
    return rows
