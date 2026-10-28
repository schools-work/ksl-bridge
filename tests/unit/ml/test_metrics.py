from ml.eval.metrics import bootstrap_ci, cer, wer


def test_wer_breakdown() -> None:
    r = wer([["A", "B", "C"]], [["A", "C", "C", "D"]])
    assert round(r["wer"], 4) == round(2 / 3, 4)
    assert cer(["abc"], ["abc"]) == 0.0


def test_bootstrap_ci_brackets_mean() -> None:
    lo, hi = bootstrap_ci([0, 1] * 50, lambda v: sum(v) / len(v), n=200)
    assert lo <= 0.5 <= hi
