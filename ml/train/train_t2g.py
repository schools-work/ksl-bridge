"""English → gloss seq2seq (same data reversed); the rule baseline + validator remain the safety net."""

from __future__ import annotations

import argparse

from .common import load_config
from .train_g2t import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/text2gloss.yaml")
    run(load_config(ap.parse_args().config), reverse=True)


if __name__ == "__main__":
    main()
