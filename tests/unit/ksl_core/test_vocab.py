import pytest

from ksl_core.vocab import BLANK, SPECIALS, UNK, Vocab


def test_build_encode_decode() -> None:
    v = Vocab.build(["WATER", "WANT", "WATER"])
    assert v.itos[: len(SPECIALS)] == SPECIALS and v.itos[BLANK] == "<blank>"
    assert v.encode(["WANT", "NOPE"]) == [v.stoi["WANT"], UNK]
    assert v.decode(v.encode(["WANT", "WATER"])) == ["WANT", "WATER"]
    assert len(v.sha256()) == 32


def test_rejects_bad_vocab() -> None:
    with pytest.raises(ValueError):
        Vocab(["WATER", "<blank>"])
