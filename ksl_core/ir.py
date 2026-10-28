"""UtteranceIR: the hub representation every modality converts to and from (Guide Part I §4.2)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Modality(str, Enum):
    SIGN = "sign"
    SPEECH = "speech"
    TEXT = "text"
    DEVICE = "device"


class Hypothesis(BaseModel):
    """One candidate reading of the input (an entry of the N-best list)."""

    glosses: list[str] = Field(default_factory=list)
    text: str | None = None
    log_prob: float
    confidence: float = Field(ge=0, le=1)


class TokenSpan(BaseModel):
    token: str
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0, le=1)
    alternatives: list[tuple[str, float]] = Field(default_factory=list)


class NonManual(BaseModel):
    question: float = Field(default=0.0, ge=0, le=1)
    negation: float = Field(default=0.0, ge=0, le=1)
    intensity: float = Field(default=0.0, ge=0, le=1)


class Intent(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    slots: dict[str, str | int | list[str]] = Field(default_factory=dict)


class Clarification(BaseModel):
    kind: str  # "choose_one" | "confirm" | "repeat" | "fingerspell"
    prompt_text: str
    prompt_glosses: list[str]
    options: list[str] = Field(default_factory=list)
    span_index: int | None = None


class UtteranceIR(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    room_id: UUID
    speaker_id: UUID
    source: Modality
    language: str = "ksl"  # "ksl" | "en" | "sw"
    created_at: datetime
    is_final: bool = False
    revision: int = 0
    best: Hypothesis
    nbest: list[Hypothesis] = Field(default_factory=list)
    spans: list[TokenSpan] = Field(default_factory=list)
    non_manual: NonManual | None = None
    intent: Intent | None = None
    needs_clarification: bool = False
    clarification: Clarification | None = None
    model_versions: dict[str, str] = Field(default_factory=dict)
