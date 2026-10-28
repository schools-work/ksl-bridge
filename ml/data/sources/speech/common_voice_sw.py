"""Mozilla Common Voice Swahili (CC0). Reads the release's TSV files (validated.tsv, clips/)."""

from __future__ import annotations

import csv
from collections.abc import Iterator

from ..base import SampleRecord, SourceLoader


class CommonVoiceSwLoader(SourceLoader):
    slug = "common-voice-sw"

    def iter_samples(self, split: str = "validated") -> Iterator[SampleRecord]:
        with open(self.root / f"{split}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                yield SampleRecord(
                    external_id=r["path"],
                    signer_id=None,
                    kind="speech",
                    glosses=[],
                    path=self.root / "clips" / r["path"],
                    text_en=None,
                    meta={
                        "text_sw": r["sentence"],
                        "speaker": r.get("client_id"),
                        "accent": r.get("accents") or r.get("accent"),
                    },
                )
