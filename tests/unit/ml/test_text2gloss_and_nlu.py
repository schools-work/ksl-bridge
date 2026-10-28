from ml.models.nlu import KeywordIntentBaseline
from ml.text2gloss.rules import rule_gloss2text, rule_text2gloss
from ml.text2gloss.validator import GlossValidator


def test_rule_text2gloss_order_and_fingerspelling() -> None:
    g, q = rule_text2gloss("Where is the bathroom today?", {"BATHROOM", "TODAY", "WHERE"}, {})
    assert g == ["TODAY", "BATHROOM", "WHERE"] and q == "wh_question"
    g, _ = rule_text2gloss("I want two burgers", {"WANT"}, {})
    assert g == ["IX-1", "WANT", "NUM:2", "FS:BURGERS"]


def test_validator_and_g2t() -> None:
    v = GlossValidator({"WATER", "ALLERGY"}, {"SODA-DRINK": "WATER"}, {"ALLERGY"}).validate(["IX-1", "ALLERGY", "XYZ"])
    assert v.glosses == ["IX-1", "ALLERGY", "FS:XYZ"] and v.critical and v.fingerspelled == 1
    assert rule_gloss2text(["IX-1", "WANT", "NUM:2", "BURGER"]) == "I want 2 burger."


def test_keyword_intent() -> None:
    r = KeywordIntentBaseline().predict("I want two chicken burger", [])
    assert r["name"] == "order_food" and r["slots"]["qty"] == 2 and r["slots"]["item"] == "chicken burger"
    assert KeywordIntentBaseline().predict("blah", [])["name"] == "unknown"
