"""Back-translation: real English domain text → synthetic glosses (t2g) → (gloss, English) pairs for g2t."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.text2gloss.rules import rule_text2gloss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("english_txt")
    ap.add_argument("--dictionary", help="file with one gloss per line (lexicon export)")
    ap.add_argument("--out", default="data/parallel/backtranslated.tsv")
    a = ap.parse_args()
    dictionary = set(Path(a.dictionary).read_text().split()) if a.dictionary else set()
    lines = [s.strip() for s in Path(a.english_txt).read_text(encoding="utf-8").splitlines() if s.strip()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in lines:
            glosses, _ = rule_text2gloss(s, dictionary, {})
            if glosses:
                f.write(f"{' '.join(glosses)}\t{s}\n")
    print(f"wrote {len(lines)} synthetic pairs → {a.out} (label them synthetic; never evaluate on them)")


if __name__ == "__main__":
    main()
