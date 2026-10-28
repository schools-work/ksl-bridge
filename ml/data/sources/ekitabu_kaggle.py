"""eKitabu KSL Video Dataset (Kaggle, CC0). Layout assumed: <root>/<CLASS>/<video>; verify after download."""

from __future__ import annotations

import re
from collections.abc import Iterator

from .base import SampleRecord, SourceLoader

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}


class EkitabuLoader(SourceLoader):
    slug = "ekitabu-ksl-video"

    def iter_samples(self) -> Iterator[SampleRecord]:
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXT:
                continue
            label = p.parent.name
            gloss = re.sub(r"[^A-Z0-9]+", "-", label.upper()).strip("-")
            m = re.search(r"(?:signer|s)[_-]?(\d+)", p.stem, re.I)
            yield SampleRecord(
                external_id=str(p.relative_to(self.root)),
                signer_id=f"ek-{m.group(1)}" if m else None,
                kind="isolated",
                glosses=[gloss],
                path=p,
                meta={"label_raw": label, "signer_inferred": m is None},
            )
