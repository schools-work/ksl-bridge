"""Envelope + typed client messages. Message names/required fields come from contracts (generated)."""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ksl_core._generated.ws_messages import BINARY_KINDS, C2S_TYPES, REQUIRED_FIELDS, ErrorCode

BIN_LANDMARKS = BINARY_KINDS["landmarks"]
BIN_AUDIO = BINARY_KINDS["audio"]
BIN_POSES = BINARY_KINDS["poses"]
BIN_TTS = BINARY_KINDS["tts"]


class Env(BaseModel):
    v: Literal[1] = 1
    type: str
    id: str | None = None
    seq: int = Field(ge=0)
    ts: int
    room: str | None = None
    ack: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Hello(BaseModel):
    type: Literal["hello"]
    token: str
    client: dict[str, str]
    resume: dict[str, Any] | None = None


class RoomJoin(BaseModel):
    type: Literal["room.join"]
    room_id: str
    role: Literal["signer", "speaker", "device", "observer"]


class TextSend(BaseModel):
    type: Literal["text.send"]
    text: Annotated[str, Field(min_length=1, max_length=2000)]
    lang: Literal["en", "sw"] = "en"


class SignStart(BaseModel):
    type: Literal["sign.start"]
    fps: float | None = None
    dominant_hand: Literal["left", "right"] | None = None


class SignStop(BaseModel):
    type: Literal["sign.stop"]


class SpeechStart(BaseModel):
    type: Literal["speech.start"]
    sample_rate: Literal[16000]
    lang_hint: Literal["en", "sw"] | None = None


class SpeechStop(BaseModel):
    type: Literal["speech.stop"]


class ClarifyAnswer(BaseModel):
    type: Literal["clarify.answer"]
    utterance_id: str
    choice: str | None = None
    retry: bool = False


class Feedback(BaseModel):
    type: Literal["feedback"]
    utterance_id: str
    kind: Literal["confirm", "correct", "reject", "clarify_choice"]
    corrected_glosses: list[str] | None = None
    corrected_text: str | None = None


class Typing(BaseModel):
    type: Literal["typing"]
    active: bool


class Ping(BaseModel):
    type: Literal["ping", "pong"]
    t: int


class AuthRefresh(BaseModel):
    type: Literal["auth.refresh"]
    token: str


class DeviceEvent(BaseModel):
    type: Literal["device.event"]
    event: str
    params: dict[str, Any] = Field(default_factory=dict)


class DeviceCapabilities(BaseModel):
    type: Literal["device.capabilities"]
    screen: bool = False
    speaker: bool = False
    camera: bool = False
    motion: bool = False


ClientMsg = TypeAdapter(
    Annotated[
        Hello
        | RoomJoin
        | TextSend
        | SignStart
        | SignStop
        | SpeechStart
        | SpeechStop
        | ClarifyAnswer
        | Feedback
        | Typing
        | Ping
        | AuthRefresh
        | DeviceEvent
        | DeviceCapabilities,
        Field(discriminator="type"),
    ]
)


class ProtocolError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def parse_client(raw: str) -> tuple[Env, Any]:
    try:
        env = Env.model_validate_json(raw)
    except ValidationError as e:
        raise ProtocolError(ErrorCode.PROTOCOL_VIOLATION, "invalid envelope") from e
    if env.type not in C2S_TYPES:
        raise ProtocolError(ErrorCode.PROTOCOL_VIOLATION, f"unknown client message type {env.type}")
    missing = [f for f in REQUIRED_FIELDS[env.type] if f not in env.data]
    if missing:
        raise ProtocolError(ErrorCode.PROTOCOL_VIOLATION, f"missing fields {missing}")
    try:
        return env, ClientMsg.validate_python({"type": env.type, **env.data})
    except ValidationError as e:
        raise ProtocolError(ErrorCode.PROTOCOL_VIOLATION, "invalid message payload") from e


def server_msg(type_: str, data: dict[str, Any], room: str | None = None, id_: str | None = None) -> dict[str, Any]:
    """Unsequenced server message; ClientConn adds seq/ts/ack when sending."""
    return {"type": type_, "room": room, "id": id_, "data": data}


def error_msg(code: ErrorCode, message: str, retry_after: float | None = None) -> dict[str, Any]:
    return server_msg("error", {"code": int(code), "message": message, "retry_after": retry_after})


def encode(env: dict[str, Any]) -> str:
    return json.dumps(env, default=str, separators=(",", ":"))


def now_ms() -> int:
    return int(time.time() * 1000)
