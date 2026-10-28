import math

from ml.decoding.ngram_lm import KneserNeyLM

SENTS = [["IX-1", "WANT", "FOOD"], ["IX-1", "WANT", "WATER"], ["IX-2", "WANT", "FOOD"], ["IX-1", "LIKE", "FOOD"]]


def test_distribution_sums_to_one_including_unk_mass() -> None:
    lm = KneserNeyLM(3).fit(SENTS)
    vocab = sorted({w for s in SENTS for w in s} | {"</s>"})
    total = sum(math.exp(lm.logp(["IX-1", "WANT"], w)) for w in vocab)
    unk = math.exp(lm.logp(["IX-1", "WANT"], "<never-seen>"))
    assert abs(total + unk - 1.0) < 1e-6


def test_context_prefers_seen_continuation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    lm = KneserNeyLM(3).fit(SENTS)
    assert lm.logp(["IX-1", "WANT"], "FOOD") > lm.logp(["IX-1", "WANT"], "FOOT")
    p = tmp_path / "lm.json"
    lm.save(p)
    lm2 = KneserNeyLM.load(p)
    assert abs(lm2.logp(["IX-1", "WANT"], "FOOD") - lm.logp(["IX-1", "WANT"], "FOOD")) < 1e-9
