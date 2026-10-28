"""Grapheme-to-phoneme. Swahili is highly phonemic → rules; English → lexicon with letter fallback."""

from __future__ import annotations

import re

SW_DIGRAPHS = ["ng'", "ch", "sh", "th", "dh", "gh", "ny", "ng", "mb", "nd", "nj", "nz", "kh"]
SW_VOWELS = set("aeiou")


def g2p_swahili(text: str) -> list[str]:
    phones: list[str] = []
    for word in re.findall(r"[a-z']+", text.lower()):
        i = 0
        while i < len(word):
            for dg in SW_DIGRAPHS:
                if word.startswith(dg, i):
                    phones.append(dg)
                    i += len(dg)
                    break
            else:
                ch = word[i]
                if ch != "'":
                    phones.append(ch)
                i += 1
        phones.append("|")  # word boundary
    return phones[:-1] if phones else phones


def g2p_english(text: str, lexicon: dict[str, list[str]] | None = None) -> list[str]:
    lexicon = lexicon or {}
    phones: list[str] = []
    for word in re.findall(r"[a-z']+", text.lower()):
        phones.extend(lexicon.get(word, list(word.replace("'", ""))))
        phones.append("|")
    return phones[:-1] if phones else phones


def g2p(text: str, lang: str) -> list[str]:
    return g2p_swahili(text) if lang == "sw" else g2p_english(text)
