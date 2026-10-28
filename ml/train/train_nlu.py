"""Intent classifier: TF-IDF + logistic regression (course baseline) on English and gloss strings."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit

from ml.models.nlu import build_tfidf_logreg

from .common import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/nlu_joint.yaml")
    cfg = load_config(ap.parse_args().config)
    rows = [json.loads(line) for line in Path(cfg["data_jsonl"]).read_text().splitlines() if line.strip()]
    X = [f"{r['text']} || {' '.join(r.get('glosses', []))}" for r in rows]
    y = [r["intent"] for r in rows]
    groups = [r.get("signer_id", i) for i, r in enumerate(rows)]
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=cfg["seed"]).split(X, y, groups))
    pipe = build_tfidf_logreg().fit([X[i] for i in tr], [y[i] for i in tr])
    print(classification_report([y[i] for i in te], pipe.predict([X[i] for i in te]), zero_division=0))
    Path(cfg["out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["out"], "wb") as f:
        pickle.dump(pipe, f)


if __name__ == "__main__":
    main()
