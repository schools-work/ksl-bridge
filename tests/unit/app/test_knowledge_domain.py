from app.modules.knowledge.domain.chunking import markdown_sections, menu_rows, recursive_split
from app.modules.knowledge.domain.grading import best_sentence, grade, key_entities, rewrite
from app.modules.knowledge.infrastructure.bm25 import BM25


def test_recursive_split_respects_size() -> None:
    text = ("Para one is here. " * 40) + "\n\nSecond para."
    chunks = recursive_split(text, 100, 20)
    assert chunks and all(len(c) <= 100 for c in chunks)


def test_markdown_sections_and_menu_rows() -> None:
    assert markdown_sections("# Menu\n## Drinks\nWater 100\n## Food\nUgali 450") == [
        ("Menu > Drinks", "Water 100"),
        ("Menu > Food", "Ugali 450"),
    ]
    assert "KSh 650" in menu_rows([{"name": "chicken burger", "price": 650}])[0]


def test_grading_rewrite_and_extraction() -> None:
    ents = key_entities("Where is the toilet?")
    assert ents == {"toilet"}
    assert not grade(ents, "The pharmacy is on floor two.", 0.9)
    assert rewrite("where is the toilet", {"toilet": ["bathroom", "choo"]}) == "where is the toilet bathroom choo"
    assert (
        best_sentence("where is the toilet", "We open at 8. The toilet is at the back.") == "The toilet is at the back."
    )


def test_bm25_ranks_relevant_first() -> None:
    bm = BM25(["water costs 100", "ugali beef 450", "pharmacy is on floor two"])
    assert bm.top("where is the pharmacy")[0][0] == 2
