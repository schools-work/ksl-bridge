"""Own recordings: read from dataset.samples (source ksl-bridge-own), approved and not excluded."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from .base import SampleRecord, SourceLoader


class OwnRecordingsLoader(SourceLoader):
    slug = "ksl-bridge-own"

    def iter_samples(self) -> Iterator[SampleRecord]:
        import sqlalchemy as sa

        url = os.environ.get("KSL_DATABASE_URL_SYNC", "postgresql://postgres:postgres@localhost:5432/ksl")
        with sa.create_engine(url).connect() as c:
            rows = c.execute(
                sa.text(
                    "SELECT s.id, s.signer_code, s.kind::text, s.glosses, s.landmarks_uri, s.text_en, s.domain::text "
                    "FROM dataset.samples s JOIN dataset.sources src ON src.id = s.source_id "
                    "WHERE src.slug = 'ksl-bridge-own' AND s.review_status = 'approved' AND s.excluded_at IS NULL"
                )
            )
            for r in rows:
                yield SampleRecord(
                    external_id=str(r[0]),
                    signer_id=r[1],
                    kind=r[2],
                    glosses=list(r[3] or []),
                    path=self.root / Path(str(r[4] or "")).name,
                    text_en=r[5],
                    meta={"domain": r[6]},
                )
