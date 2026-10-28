"""Retention classes (values live in privacy.retention_policies; this is the list of classes we rely on)."""

from __future__ import annotations

DATA_CLASSES = (
    "realtime_landmarks",
    "transcripts",
    "inference_logs",
    "feedback_clips",
    "dataset_samples",
    "audit_events",
    "ws_sessions",
)
