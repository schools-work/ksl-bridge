"""Pick each gloss's dictionary clip: the medoid (min total DTW distance) across signers (Guide Part IV §11.1)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ksl_core.normalize import build_features, normalize_sequence
from ml.models.baselines import dtw_distance
from ml.models.production import trim_rest

from .common import read_manifest


def medoid(clips: list[np.ndarray], window: int = 20) -> tuple[int, list[float]]:
    feats = [build_features(c)[0][:, 642:888] for c in clips]
    n = len(feats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_distance(feats[i], feats[j], window)
    costs = D.sum(1)
    return int(costs.argmin()), costs.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="artifacts/sign_dict")
    a = ap.parse_args()
    groups: dict[str, list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for it in read_manifest(a.manifest):
        if len(it["glosses"]) == 1:
            groups[it["glosses"][0]].append(it)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for gloss, items in groups.items():
        clips = [
            trim_rest(
                normalize_sequence(
                    np.load(i["landmarks_path"]).astype(np.float32), mirror=i.get("dominant_hand") == "left"
                )
            )
            for i in items
        ]
        k, costs = medoid(clips) if len(clips) > 1 else (0, [0.0])
        np.save(out / f"{gloss}.npy", clips[k].astype(np.float16))
        report[gloss] = {"sample_id": items[k].get("sample_id"), "n": len(items), "cost": costs[k]}
    (out / "medoids.json").write_text(json.dumps(report, indent=2))
    print(f"{len(report)} medoids → {out} (Deaf reviewers approve before use)")


if __name__ == "__main__":
    main()
