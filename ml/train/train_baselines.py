"""Week-one baselines: k-NN + DTW and Random Forest on a manifest (signer-independent splits)."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from ksl_core.normalize import build_features, normalize_sequence
from ml.models.baselines import KnnDTW, random_forest, summary_features

from .common import by_split, load_config, read_manifest, seed_everything


def _feats(item: dict) -> np.ndarray:  # type: ignore[type-arg]
    xyz = np.load(item["landmarks_path"]).astype(np.float32)
    f, _ = build_features(normalize_sequence(xyz, mirror=item.get("dominant_hand") == "left"))
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/baselines.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg["seed"])
    splits = by_split(read_manifest(cfg["manifest"]))
    Xtr = [_feats(i) for i in splits["train"]]
    ytr = [i["glosses"][0] for i in splits["train"]]
    Xte = [_feats(i) for i in splits["test"]]
    yte = [i["glosses"][0] for i in splits["test"]]
    print(
        f"train {len(Xtr)} / test {len(Xte)}; classes {len(set(ytr))}; majority baseline "
        f"{Counter(yte).most_common(1)[0][1] / max(1, len(yte)):.3f}"
    )
    knn = KnnDTW(k=cfg["knn"]["k"], window=cfg["knn"]["window"], pca_dim=cfg["knn"]["pca_dim"]).fit(Xtr, ytr)
    acc_knn = float(np.mean([knn.predict(x) == y for x, y in zip(Xte, yte, strict=True)]))
    rf = random_forest(cfg["rf"]["n_estimators"], cfg["seed"]).fit(np.stack([summary_features(x) for x in Xtr]), ytr)
    acc_rf = float(np.mean(rf.predict(np.stack([summary_features(x) for x in Xte])) == np.array(yte)))
    print(f"k-NN+DTW top-1: {acc_knn:.3f}   RandomForest top-1: {acc_rf:.3f}")


if __name__ == "__main__":
    main()
