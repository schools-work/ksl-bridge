"""Maseno KSL Word-Based Pose Dataset (Zenodo 14974973). RESTRICTED until access is granted."""

from __future__ import annotations

from collections.abc import Iterator

from ksl_core.errors import SourceUnavailable

from .base import SampleRecord, SourceLoader, check_allowed


class MasenoPoseLoader(SourceLoader):
    slug = "maseno-ksl-pose"

    def iter_samples(self) -> Iterator[SampleRecord]:
        check_allowed(self.slug, "train")  # raises while dataset.sources.usage = 'blocked'
        raise SourceUnavailable(
            "Maseno pose files: implement the parser once the access-granted file format is known "
            "(update dataset.sources.licence/usage at the same time)"
        )
