"""PipelineSession: per-connection orchestration. Holds buffers/state; delegates each path to a use case."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import numpy as np

from app.platform.principal import Principal
from app.settings import get_settings
from ksl_core.landmarks import LandmarkBatch
from ksl_core.normalize import StreamingNormalizer

from ..domain.context import DialogueContext
from ..domain.endpointing import Endpointer
from ..domain.ports import Broadcaster, JobClient
from ..domain.session_state import SignStateMachine

MAX_AUDIO_BYTES = 16000 * 2 * 30  # 30 s of PCM16 mono


class PipelineSession:
    def __init__(
        self,
        session_id: str,
        principal: Principal,
        broadcaster: Broadcaster,
        jobs: JobClient,
        reply: Callable[..., Any],
    ):
        s = get_settings()
        self.session_id, self.principal = session_id, principal
        self.bus, self.jobs, self.reply = broadcaster, jobs, reply
        self.room_id: UUID | None = None
        self.member_id: str | None = None
        self.role = "observer"
        self.modality_out = "text"
        self.room_info: dict[str, Any] = {}
        self.context = DialogueContext()
        self.sign = SignStateMachine()
        self.normalizer = StreamingNormalizer()
        self.endpointer = Endpointer(still_ms=s.endpoint_still_ms)
        self.frames: list[np.ndarray] = []  # normalised frames of the current utterance
        self.mirror = False
        self.fps = 30.0
        self.frames_since_partial = 0
        self.current_utterance_id: str | None = None
        self.audio = bytearray()
        self.speech_lang: str | None = None
        self.pending_clarifications: dict[str, dict[str, Any]] = {}
        self.max_frames = int(s.max_utterance_seconds * 30)

    async def attach(
        self, room_id: UUID, member_id: str, role: str, modality_out: str, room_info: dict[str, Any]
    ) -> None:
        from . import on_text

        self.room_id, self.member_id, self.role, self.modality_out = room_id, member_id, role, modality_out
        self.room_info = room_info
        self.context.venue_type = str(room_info.get("context_profile") or "general")
        await on_text.load_venue_context(self)

    def set_signing_prefs(self, fps: float | None, dominant_hand: str | None) -> None:
        if fps:
            self.fps = float(fps)
        self.mirror = dominant_hand == "left"

    async def on_landmarks(self, batch: LandmarkBatch) -> None:
        from . import on_landmarks

        await on_landmarks.handle(self, batch)

    async def end_signing(self) -> None:
        from . import finalise_utterance

        await finalise_utterance.finalise(self)

    async def on_text(self, text: str, lang: str) -> None:
        from . import on_text

        await on_text.handle(self, text, lang)

    def start_speech(self, lang_hint: str | None) -> None:
        self.audio.clear()
        self.speech_lang = lang_hint

    async def on_audio(self, pcm: bytes) -> None:
        if len(self.audio) + len(pcm) <= MAX_AUDIO_BYTES:
            self.audio.extend(pcm)

    async def end_speech(self) -> None:
        from . import on_audio

        await on_audio.handle(self)

    async def on_clarify_answer(self, utterance_id: str, choice: str | None, retry: bool) -> None:
        from . import finalise_utterance

        await finalise_utterance.resolve_clarification(self, utterance_id, choice, retry)

    async def on_device_message(self, msg: dict[str, Any]) -> None:
        from . import on_text

        await on_text.handle_device(self, msg)

    async def close(self) -> None:
        self.frames.clear()
        self.audio.clear()
