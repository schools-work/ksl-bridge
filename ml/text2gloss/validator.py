"""Every produced gloss must exist in the dictionary; synonym lookup, then fingerspelling fallback."""

from __future__ import annotations

from dataclasses import dataclass

STRUCTURAL = {"IX-1", "IX-2", "IX-3", "IX-1-PL", "IX-3-PL", "POSS-1", "POSS-2", "NOT"}


@dataclass
class Validated:
    glosses: list[str]
    fingerspelled: int
    critical: bool


class GlossValidator:
    def __init__(self, dictionary: set[str], synonyms: dict[str, str], critical: set[str]):
        self.dictionary, self.synonyms, self.critical = dictionary, synonyms, critical

    def validate(self, glosses: list[str]) -> Validated:
        out: list[str] = []
        fs = 0
        for g in glosses:
            if g.startswith(("FS:", "NUM:")) or g in STRUCTURAL or g in self.dictionary or not self.dictionary:
                out.append(g)
                fs += g.startswith("FS:")
            elif g in self.synonyms:
                out.append(self.synonyms[g])
            else:
                out.append(f"FS:{g.replace('-', '')}")
                fs += 1
        return Validated(out, fs, any(g in self.critical for g in out))
