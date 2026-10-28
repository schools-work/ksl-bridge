"""Rule baselines for Text→Gloss and Gloss→Text. PLACEHOLDER GRAMMAR: validate with Deaf KSL consultants."""

from __future__ import annotations

import re

DROP = {"a", "an", "the", "is", "are", "am", "was", "were", "be", "to", "of", "do", "does", "did", "please"}
WH = {"what", "where", "when", "who", "why", "how", "which"}
TIME = {"today", "tomorrow", "yesterday", "now", "later", "morning", "evening", "tonight"}
PRON = {
    "i": "IX-1",
    "me": "IX-1",
    "my": "POSS-1",
    "you": "IX-2",
    "your": "POSS-2",
    "he": "IX-3",
    "she": "IX-3",
    "they": "IX-3-PL",
    "we": "IX-1-PL",
}
NEG = {"not", "no", "never", "don't", "dont"}
NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def simple_lemma(w: str) -> str:
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def rule_text2gloss(text: str, dictionary: set[str], synonyms: dict[str, str]) -> tuple[list[str], str | None]:
    """Returns (glosses, question_type). Heuristic order: TIME first, body, NEG, WH last."""
    raw = text.strip()
    words = re.findall(r"[a-zA-Z']+|\d+", raw.lower())
    time_part: list[str] = []
    body: list[str] = []
    wh: list[str] = []
    neg = False
    for w in words:
        if w in DROP:
            continue
        if w in NEG:
            neg = True
            continue
        if w in PRON:
            body.append(PRON[w])
            continue
        if w.isdigit() or w in NUM_WORDS:
            body.append(f"NUM:{w if w.isdigit() else NUM_WORDS[w]}")
            continue
        g = simple_lemma(w).upper().replace("'", "")
        g = synonyms.get(g, synonyms.get(w.upper(), g))
        target = time_part if w in TIME else wh if w in WH else body
        target.append(g if (g in dictionary or not dictionary) else f"FS:{w.upper()}")
    question = "wh_question" if wh else ("yes_no_question" if raw.endswith("?") else None)
    return time_part + body + (["NOT"] if neg else []) + wh, question


PRON_EN = {
    "IX-1": "I",
    "IX-2": "you",
    "IX-3": "they",
    "IX-1-PL": "we",
    "IX-3-PL": "they",
    "POSS-1": "my",
    "POSS-2": "your",
}


def rule_gloss2text(glosses: list[str]) -> str:
    """Very small fallback realiser (used only when no gloss2text model is deployed)."""
    words: list[str] = []
    for g in glosses:
        if g in PRON_EN:
            words.append(PRON_EN[g])
        elif g.startswith("NUM:"):
            words.append(g[4:])
        elif g.startswith("FS:"):
            words.append(g[3:].title())
        elif g == "NOT":
            words.append("not")
        else:
            words.append(g.replace("-", " ").lower())
    s = " ".join(words).strip()
    return (s[:1].upper() + s[1:] + ".") if s else ""
