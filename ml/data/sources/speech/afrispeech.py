"""AfriSpeech-200 (CC BY-NC-SA 4.0): EVAL ONLY. Kenyan-accented English subsets."""

from __future__ import annotations

import csv
from collections.abc import Iterator

from ..base import SampleRecord, SourceLoader, check_allowed

KENYAN_ACCENTS = {"swahili", "kiswahili", "kikuyu", "luhya", "luo"}


class AfriSpeechLoader(SourceLoader):
    slug = "afrispeech-200"

    def iter_samples(self, split: str = "test") -> Iterator[SampleRecord]:
        check_allowed(self.slug, "eval")
        for tsv in self.root.rglob(f"*{split}*.csv"):
            with open(tsv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if (r.get("accent") or "").lower() not in KENYAN_ACCENTS:
                        continue
                    yield SampleRecord(
                        external_id=r["audio_id"],
                        signer_id=None,
                        kind="speech",
                        glosses=[],
                        path=self.root / r["audio_paths"].lstrip("/"),
                        text_en=r["transcript"],
                        meta={"accent": r["accent"], "speaker": r.get("user_ids"), "eval_only": True},
                    )
