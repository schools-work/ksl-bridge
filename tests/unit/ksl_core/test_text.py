from ksl_core.text import normalize_text, words_to_number


def test_normalize() -> None:
    assert normalize_text("I don't want KSh 1,500 sasa!") == "i do not want 1500 shillings hello"


def test_numbers_en_sw() -> None:
    assert words_to_number("two hundred and fifty".split()) == 250
    assert words_to_number("five thousand".split()) == 5000
    assert words_to_number("mia mbili".split(), "sw") == 200
    assert words_to_number("elfu mbili mia tano".split(), "sw") == 2500
    assert words_to_number("kumi na mbili".split(), "sw") == 12
    assert words_to_number("hello".split()) is None
