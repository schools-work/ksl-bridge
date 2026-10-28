"""Consent scopes and what each one unlocks (single home for the meaning of a scope)."""

from __future__ import annotations

SCOPES = {
    "store_transcripts": "Room text may be stored for 30 days",
    "contribute_landmarks": "Body landmarks (not video) may train models",
    "contribute_video": "Signing video may be stored and train models",
    "contribute_audio": "Voice recordings may train speech models",
    "personalisation": "The system may learn your personal signing style",
    "research_publication": "Anonymised results may be published",
    "demographics": "Demographics may be used only for fairness checks",
}
CONTRIBUTION_SCOPES = frozenset({"contribute_landmarks", "contribute_video", "contribute_audio"})


def validate_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"unknown consent scope {scope}")
    return scope
