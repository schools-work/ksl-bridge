"""Shared error types (framework-free)."""

from __future__ import annotations


class KslError(Exception):
    """Base class for all project errors."""

    code: str = "ksl_error"


class InvalidFrame(KslError):
    """A binary landmark/audio frame failed validation (bad magic, version, size or values)."""

    code = "invalid_frame"


class SpecVersionMismatch(InvalidFrame):
    code = "spec_version_mismatch"


class VocabularyMismatch(KslError):
    """A model artefact's vocabulary hash doesn't match the registry."""

    code = "vocabulary_mismatch"


class SourceUnavailable(KslError):
    """A dataset source is blocked, restricted or not yet downloaded."""

    code = "source_unavailable"


class LicenceViolation(KslError):
    """An operation would use data outside its licence (see dataset.sources.usage)."""

    code = "licence_violation"


class DecisionRequired(KslError):
    """Code depends on a PROVISIONAL group decision (MASTER_PROMPT §4.1 / §12.2)."""

    code = "decision_required"
