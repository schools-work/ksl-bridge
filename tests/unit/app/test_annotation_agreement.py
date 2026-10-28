from app.modules.annotation.application.agreement import cohen_kappa, gloss_edit_agreement


def test_kappa_and_edit_agreement() -> None:
    assert cohen_kappa(["q", "q", "n", "n"], ["q", "q", "n", "n"]) == 1.0
    assert cohen_kappa(["q", "n", "q", "n"], ["n", "q", "n", "q"]) < 0
    assert gloss_edit_agreement([(["A", "B"], ["A", "B"])]) == 1.0
    assert 0 < gloss_edit_agreement([(["A", "B", "C"], ["A", "C"])]) < 1
