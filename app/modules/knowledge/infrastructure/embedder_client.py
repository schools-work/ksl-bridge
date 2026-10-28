"""Query embeddings via the embed worker (our sentence encoder). None ⇒ keyword-only retrieval."""

from __future__ import annotations

from typing import Any


class EmbedderClient:
    def __init__(self, jobs: Any):
        self.jobs = jobs

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        out = await self.jobs.call("embed", {"texts": texts}, deadline_ms=1500)
        if not out or not out.get("vectors"):
            return [None] * len(texts)
        return list(out["vectors"])
