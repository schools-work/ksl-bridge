"""Sentence-embedding worker for RAG (our contrastive encoder). No model ⇒ keyword-only retrieval."""

from __future__ import annotations

from typing import Any

from .adapters import ModelAdapter, ModelInfo, ModelUnavailable, register
from .loader import fetch_artifact, live_model
from .runner import main


@register("embed")
class EmbedAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("embedder", "unloaded", "-", "cpu")
        self.model: Any = None

    def load(self) -> None:
        m = live_model("embedder")
        if m is None:
            return
        from ml.models.embedder import SentenceEncoder

        self.model = SentenceEncoder.from_checkpoint(fetch_artifact(m))
        self.info = ModelInfo("embedder", m.name, m.version, "cpu", m.is_bootstrap)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise ModelUnavailable("no embedder deployed")
        return [{"vectors": self.model.encode(i["texts"]).tolist()} for i in inputs]


if __name__ == "__main__":
    main("embed", EmbedAdapter())
