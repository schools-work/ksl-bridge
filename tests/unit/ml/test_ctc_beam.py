import math

import numpy as np

from ml.decoding.alignment import ctc_force_align, token_spans
from ml.decoding.ctc_beam import CTCPrefixBeamSearch, greedy_decode

ITOS = ["<blank>", "<pad>", "<unk>", "<bos>", "<eos>", "A", "B", "C"]


def test_prefix_probabilities_sum_to_one_without_pruning() -> None:
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(8), size=6))
    lp[:, [1, 3, 4]] = -np.inf  # specials never emitted
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    d = CTCPrefixBeamSearch(ITOS, beam=10**6, beta=0.0, prune_logp=-1e9)
    d.step(lp)
    assert abs(sum(math.exp(b.acoustic()) for b in d.beams.values()) - 1.0) < 1e-9


def test_peaked_posteriors_equal_greedy() -> None:
    rng = np.random.default_rng(3)
    ids = rng.choice([0, 5, 6, 7], size=40)
    lp = np.full((40, 8), -9.0)
    lp[np.arange(40), ids] = -0.001
    d = CTCPrefixBeamSearch(ITOS, beam=4, beta=0.0)
    d.step(lp)
    assert d.nbest(1)[0][0] == greedy_decode(lp, ITOS)


def test_streaming_equals_single_pass_and_bias_helps() -> None:
    rng = np.random.default_rng(1)
    lp = np.log(rng.dirichlet(np.ones(8) * 0.3, size=20))
    a = CTCPrefixBeamSearch(ITOS, beam=8)
    a.step(lp)
    b = CTCPrefixBeamSearch(ITOS, beam=8)
    b.step(lp[:9])
    b.step(lp[9:])
    assert a.nbest(3) == b.nbest(3)
    biased = CTCPrefixBeamSearch(ITOS, beam=8, context_bias={"C": 2.0})
    biased.step(lp)
    assert biased.nbest(1)[0][0].count("C") >= a.nbest(1)[0][0].count("C")
    assert not any(t in ("<pad>", "<bos>", "<eos>") for t in a.nbest(1)[0][0])


def test_force_align_and_spans() -> None:
    lp = np.full((6, 8), -9.0)
    for t, c in enumerate([5, 5, 0, 6, 6, 0]):
        lp[t, c] = -0.01
    assert ctc_force_align(lp, [5, 6]) == [(0, 1), (3, 4)]
    spans = token_spans(lp, ITOS, [(["A", "B"], -1.0), (["A", "C"], -2.0)], frame_ms=100)
    assert [s["token"] for s in spans] == ["A", "B"]
    assert spans[1]["alternatives"][0][0] == "C"
