"""ASR worker. Our Conformer-CTC when deployed; otherwise a LOCAL bootstrap model (ADR-0003, provisional)."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from ksl_core.errors import DecisionRequired

from .adapters import ModelAdapter, ModelInfo, ModelUnavailable, register
from .loader import fetch_artifact, live_model
from .runner import main


@register("asr")
class ASRAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("asr", "unloaded", "-", "cpu")
        self.impl: Any = None

    def load(self) -> None:
        m = live_model("asr")
        if m is None:
            return
        if m.is_bootstrap:
            self.impl = _BootstrapASR(m.name)
        else:
            from ml.models.asr import ConformerCTCRecognizer

            self.impl = ConformerCTCRecognizer.from_checkpoint(fetch_artifact(m))
        self.info = ModelInfo("asr", m.name, m.version, "cpu", m.is_bootstrap)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.impl is None:
            raise ModelUnavailable("no ASR model deployed")
        outs = []
        for i in inputs:
            pcm = np.frombuffer(base64.b64decode(i["pcm16_b64"]), dtype="<i2").astype(np.float32) / 32768.0
            text, lang, conf = self.impl.transcribe(pcm, i.get("lang_hint"))
            outs.append({"text": text, "lang": lang, "confidence": conf, "model_versions": {"asr": self.info.version}})
        return outs


class _BootstrapASR:
    """Runs a locally installed open-weights recogniser (e.g. faster-whisper). Never a hosted API."""

    def __init__(self, name: str):
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise DecisionRequired("bootstrap ASR selected but no local engine installed; see ADR-0003") from e
        self.model = WhisperModel(
            name.replace("asr_bootstrap_", "").replace("_", "-") or "small", device="cpu", compute_type="int8"
        )

    def transcribe(self, pcm: np.ndarray, lang_hint: str | None) -> tuple[str, str, float]:
        segments, info = self.model.transcribe(pcm, language=lang_hint, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments)
        return text, info.language or (lang_hint or "en"), float(info.language_probability or 0.8)


if __name__ == "__main__":
    main("asr", ASRAdapter())
