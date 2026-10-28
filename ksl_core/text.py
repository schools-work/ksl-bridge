"""English/Swahili text normalisation shared by ASR, TTS, Text→Gloss and NLU."""

from __future__ import annotations

import re
import unicodedata

_EN_NUM = {
    "zero": 0,
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
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "hundred": 100,
    "thousand": 1000,
}
_SW_NUM = {
    "sifuri": 0,
    "moja": 1,
    "mbili": 2,
    "tatu": 3,
    "nne": 4,
    "tano": 5,
    "sita": 6,
    "saba": 7,
    "nane": 8,
    "tisa": 9,
    "kumi": 10,
    "ishirini": 20,
    "thelathini": 30,
    "arobaini": 40,
    "hamsini": 50,
    "mia": 100,
    "elfu": 1000,
}
_CONTRACTIONS = {
    "don't": "do not",
    "can't": "can not",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "i'd": "i would",
    "i'll": "i will",
    "isn't": "is not",
    "doesn't": "does not",
    "what's": "what is",
    "where's": "where is",
    "you're": "you are",
}
# Common Sheng / colloquial forms → standard words. Extend with community input; never guess meanings.
SHENG = {
    "maze": "friend",
    "fare": "fare",
    "doh": "money",
    "mbao": "twenty",
    "ashuu": "fifty",
    "soo": "hundred",
    "ngiri": "thousand",
    "fiti": "fine",
    "sasa": "hello",
    "poa": "fine",
}

_KSH = re.compile(r"\b(?:ksh|kshs|kes)\.?\s*([\d,]+)\b", re.I)
_WS = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize_text(text: str, lang: str = "en", expand_sheng: bool = True) -> str:
    """Lower-case, expand contractions and currency, map Sheng, collapse whitespace."""
    s = strip_accents(text).lower().replace("’", "'")
    s = _KSH.sub(lambda m: f"{m.group(1).replace(',', '')} shillings", s)
    if lang == "en":
        for k, v in _CONTRACTIONS.items():
            s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    s = re.sub(r"[^\w\s'-]", " ", s)
    if expand_sheng:
        s = " ".join(SHENG.get(w, w) for w in s.split())
    return _WS.sub(" ", s).strip()


def _sw_number(tokens: list[str]) -> int | None:
    """Swahili: the multiplier FOLLOWS mia/elfu ('mia mbili' = 200, 'elfu mbili mia tano' = 2500)."""
    total, i, seen = 0, 0, False
    while i < len(tokens):
        t = tokens[i]
        if t == "na":
            i += 1
            continue
        if t in ("mia", "elfu"):
            mult = 100 if t == "mia" else 1000
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is not None and nxt in _SW_NUM and nxt not in ("mia", "elfu"):
                total += mult * _SW_NUM[nxt]
                i += 2
            else:
                total += mult
                i += 1
            seen = True
            continue
        if t.isdigit():
            total += int(t)
        elif t in _SW_NUM:
            total += _SW_NUM[t]
        else:
            return None
        seen = True
        i += 1
    return total if seen else None


def _en_number(tokens: list[str]) -> int | None:
    """English: 'two hundred and fifty' = 250, 'five thousand' = 5000."""
    total, current, seen = 0, 0, False
    for t in tokens:
        if t == "and":
            continue
        if t.isdigit():
            current += int(t)
        elif t in _EN_NUM:
            v = _EN_NUM[t]
            if v == 100:
                current = max(current, 1) * 100
            elif v == 1000:
                total += max(current, 1) * 1000
                current = 0
            else:
                current += v
        else:
            return None
        seen = True
    return total + current if seen else None


def words_to_number(tokens: list[str], lang: str = "en") -> int | None:
    """Parse a number phrase; returns None if any token isn't part of a number."""
    return _sw_number(tokens) if lang == "sw" else _en_number(tokens)
