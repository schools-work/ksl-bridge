"""Build Kneser–Ney gloss LMs (general + per-domain) from gloss sentences (one sentence per line)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.decoding.ngram_lm import KneserNeyLM

from .common import load_config


def read_sentences(path: str | Path) -> list[list[str]]:
    return [line.split() for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/gloss_lm.yaml")
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, corpus in cfg["corpora"].items():
        sents = read_sentences(corpus["train"])
        lm = KneserNeyLM(cfg["order"], cfg["discount"]).fit(sents)
        held = read_sentences(corpus["heldout"]) if corpus.get("heldout") else sents[-max(1, len(sents) // 10) :]
        lm.save(out / f"gloss_kn{cfg['order']}_{name}.json")
        print(f"{name}: {len(sents)} sentences, held-out perplexity {lm.perplexity(held):.2f}")


if __name__ == "__main__":
    main()
