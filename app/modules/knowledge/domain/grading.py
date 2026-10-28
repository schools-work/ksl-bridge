"""Relevance grading + query rewrite (fixes for naive-RAG failure modes 1–2). Pure."""

from __future__ import annotations

import re

STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "where",
    "what",
    "how",
    "do",
    "you",
    "have",
    "i",
    "can",
    "my",
    "to",
    "of",
    "please",
    "much",
    "does",
    "it",
    "there",
    "any",
}


def key_entities(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOP and len(w) > 2}


def grade(question_entities: set[str], chunk: str, score: float, min_score: float = 0.35) -> bool:
    text = chunk.lower()
    entity_hit = (not question_entities) or any(e in text for e in question_entities)
    return score >= min_score and entity_hit


def rewrite(question: str, synonyms: dict[str, list[str]]) -> str | None:
    """Expand words with dictionary synonyms (toilet → bathroom washroom choo). None if nothing to add."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    extra = [s for w in words for s in synonyms.get(w, []) if s not in words]
    return f"{question} {' '.join(extra)}" if extra else None


def best_sentence(question: str, chunk: str) -> str:
    """Extractive answer: the sentence with the most question-entity overlap."""
    ents = key_entities(question)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    if not sentences:
        return chunk.strip()
    return max(sentences, key=lambda s: sum(e in s.lower() for e in ents))
