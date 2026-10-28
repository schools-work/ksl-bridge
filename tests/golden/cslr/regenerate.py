"""Regenerate cases.json from fixed seeds (reviewers check the diff)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.decoding.ctc_beam import CTCPrefixBeamSearch

ITOS = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>", "IX-1", "WANT", "WATER", "WAIT", "FOOD"]


def build() -> list[dict]:  # type: ignore[type-arg]
    cases = []
    for name, seed, beam, beta in (("small_beam", 1, 4, 1.0), ("wide_beam", 2, 16, 0.5), ("no_bonus", 3, 8, 0.0)):
        rng = np.random.default_rng(seed)
        lp = np.log(rng.dirichlet(np.ones(len(ITOS)) * 0.4, size=18)).round(6)
        d = CTCPrefixBeamSearch(ITOS, beam=beam, beta=beta)
        d.step(lp)
        cases.append(
            {
                "name": name,
                "itos": ITOS,
                "beam": beam,
                "beta": beta,
                "log_probs": lp.tolist(),
                "expected": [[g, round(s, 6)] for g, s in d.nbest(3)],
            }
        )
    return cases


if __name__ == "__main__":
    out = Path(__file__).with_name("cases.json")
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out}")
