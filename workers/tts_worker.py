"""TTS worker. Our FastSpeech2+HiFi-GAN when deployed; otherwise a LOCAL bootstrap voice (ADR-0003)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ksl_core.errors import DecisionRequired
from ksl_core.text import normalize_text

from .adapters import ModelAdapter, ModelInfo, ModelUnavailable, register
from .loader import fetch_artifact, live_model
from .runner import main


@register("tts")
class TTSAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("tts", "unloaded", "-", "cpu")
        self.impl: Any = None

    def load(self) -> None:
        m = live_model("tts")
        if m is None:
            return
        if m.is_bootstrap:
            self.impl = _BootstrapTTS(str(fetch_artifact(m)))
        else:
            from ml.models.tts import Synthesizer

            self.impl = Synthesizer.from_checkpoint(fetch_artifact(m))
        self.info = ModelInfo("tts", m.name, m.version, "cpu", m.is_bootstrap)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.impl is None:
            raise ModelUnavailable("no TTS model deployed")
        outs = []
        for i in inputs:
            wav, sr = self.impl.synthesize(
                normalize_text(i["text"], i.get("lang", "en"), expand_sheng=False), i.get("lang", "en")
            )
            pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
            outs.append({"pcm16_b64": base64.b64encode(pcm).decode(), "sample_rate": sr})
        return outs


class _BootstrapTTS:
    def __init__(self, voice_path: str):
        try:
            from piper import PiperVoice  # type: ignore[import-not-found]
        except ImportError as e:
            raise DecisionRequired("bootstrap TTS selected but piper is not installed; see ADR-0003") from e
        self.voice = PiperVoice.load(voice_path)

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        audio = b"".join(self.voice.synthesize_stream_raw(text))
        return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0, int(self.voice.config.sample_rate)


if __name__ == "__main__":
    main("tts", TTSAdapter())
