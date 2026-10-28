"""AfriVoices-KE (CC BY 4.0; request form). Kenyan languages; HF org Anv-ke."""

from __future__ import annotations

from collections.abc import Iterator

from ..base import SampleRecord, SourceLoader


class AfriVoicesKeLoader(SourceLoader):
    slug = "afrivoices-ke"

    def iter_samples(self) -> Iterator[SampleRecord]:
        import json

        for meta in self.root.rglob("*.jsonl"):
            for line in meta.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                yield SampleRecord(
                    external_id=r.get("id", r.get("audio")),
                    signer_id=None,
                    kind="speech",
                    glosses=[],
                    path=meta.parent / r["audio"],
                    meta={"text": r.get("text"), "language": r.get("language")},
                )
