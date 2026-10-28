"""NLU: keyword baseline (no training), TF-IDF + logistic regression (course), joint model hook."""

from __future__ import annotations

import pickle
import re
from typing import Any

INTENT_KEYWORDS = {
    "order_food": {
        "want",
        "order",
        "burger",
        "chips",
        "chapati",
        "ugali",
        "food",
        "eat",
        "water",
        "soda",
        "tea",
        "juice",
        "drink",
    },
    "ask_location": {"where", "toilet", "bathroom", "pharmacy", "office", "exit", "reception"},
    "pay": {"pay", "bill", "mpesa", "m-pesa", "cash", "card"},
    "book_appointment": {"appointment", "doctor", "book", "see"},
    "greet": {"hello", "hi", "habari", "sasa", "good"},
    "help": {"help", "emergency", "interpreter"},
}
ITEMS = ["chicken burger", "burger", "chips", "chapati", "ugali beef", "ugali", "water", "soda", "tea", "juice"]
PLACES = ["toilet", "bathroom", "pharmacy", "reception", "exit", "office"]
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def extract_slots(text: str, glosses: list[str]) -> dict[str, Any]:
    t = text.lower()
    slots: dict[str, Any] = {}
    item = next((i for i in ITEMS if i in t), None)
    if item:
        slots["item"] = item
    place = next((p for p in PLACES if p in t), None)
    if place:
        slots["place"] = place
    m = re.search(r"\b(\d+)\b", t)
    qty = int(m.group(1)) if m else next((v for w, v in NUM.items() if re.search(rf"\b{w}\b", t)), None)
    for g in glosses:
        if g.startswith("NUM:") and g[4:].isdigit():
            qty = int(g[4:])
    if qty:
        slots["qty"] = qty
    if "mpesa" in t or "m-pesa" in t:
        slots["method"] = "mpesa"
    elif "cash" in t:
        slots["method"] = "cash"
    return slots


class KeywordIntentBaseline:
    """Transparent rule baseline. Report it beside the learned models; never present it as 'AI'."""

    def predict(self, text: str, glosses: list[str]) -> dict[str, Any]:
        words = set(re.findall(r"[a-z\-]+", text.lower())) | {g.lower() for g in glosses}
        scores = {k: len(words & v) for k, v in INTENT_KEYWORDS.items()}
        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        if scores[best] == 0:
            return {"name": "unknown", "confidence": 0.0, "slots": {}}
        return {
            "name": best,
            "confidence": round(min(0.95, 0.4 + 0.5 * scores[best] / max(total, 1)), 3),
            "slots": extract_slots(text, glosses),
        }


def build_tfidf_logreg() -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2),
    )
    return make_pipeline(feats, LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))


class SklearnIntentModel:
    def __init__(self, pipeline: Any):
        self.p = pipeline

    def predict(self, text: str, glosses: list[str]) -> dict[str, Any]:
        x = f"{text} || {' '.join(glosses)}"
        proba = self.p.predict_proba([x])[0]
        i = int(proba.argmax())
        return {"name": str(self.p.classes_[i]), "confidence": float(proba[i]), "slots": extract_slots(text, glosses)}


def load_sklearn_pipeline(path: Any) -> SklearnIntentModel:
    with open(path, "rb") as f:
        return SklearnIntentModel(pickle.load(f))
