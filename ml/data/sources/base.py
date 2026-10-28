"""SourceLoader protocol + licence gate (dataset.sources.usage)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ksl_core.errors import LicenceViolation, SourceUnavailable

USAGE_ALLOWED = {
    "train": {"train", "pretrain", "eval"},
    "pretrain_only": {"pretrain"},
    "eval_only": {"eval"},
    "blocked": set(),
}


@dataclass
class SampleRecord:
    external_id: str
    signer_id: str | None
    kind: str
    glosses: list[str]
    path: Path
    text_en: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def source_usage(slug: str) -> str:
    import sqlalchemy as sa

    url = os.environ.get("KSL_DATABASE_URL_SYNC", "postgresql://postgres:postgres@localhost:5432/ksl")
    with sa.create_engine(url).connect() as c:
        row = c.execute(sa.text("SELECT usage FROM dataset.sources WHERE slug = :s"), {"s": slug}).first()
    if row is None:
        raise SourceUnavailable(f"source {slug} is not registered in dataset.sources")
    return str(row[0])


def check_allowed(slug: str, purpose: str, usage: str | None = None) -> None:
    usage = usage or source_usage(slug)
    if purpose not in USAGE_ALLOWED[usage]:
        raise LicenceViolation(f"source {slug} has usage={usage}; '{purpose}' is not allowed")


class SourceLoader:
    slug: str = ""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise SourceUnavailable(f"{self.slug}: {self.root} not found; run tools/scripts/download_sources.sh")

    def iter_samples(self) -> Iterator[SampleRecord]:
        raise NotImplementedError
