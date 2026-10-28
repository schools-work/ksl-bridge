"""RQ4 ablation: no LM vs LM vs LM + venue bias, on the same test set."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ml.decoding.ctc_beam import CTCPrefixBeamSearch

from .metrics import wer


def ablate(
    logprob_fn: Callable[[np.ndarray], np.ndarray],
    items: list[dict[str, Any]],
    itos: list[str],
    lm: Any,
    bias_for: Callable[[dict[str, Any]], dict[str, float]],
    beam: int = 8,
) -> dict[str, float]:
    settings = {"no_lm": (None, False), "lm": (lm, False), "lm_bias": (lm, True)}
    out = {}
    for name, (use_lm, use_bias) in settings.items():
        hyps = []
        for it in items:
            dec = CTCPrefixBeamSearch(itos, lm=use_lm, beam=beam, context_bias=bias_for(it) if use_bias else {})
            dec.step(logprob_fn(it["xyz"]))
            nb = dec.nbest(1, final=True)
            hyps.append(nb[0][0] if nb else [])
        out[name] = wer([it["glosses"] for it in items], hyps)["wer"]
    return out
