"""structlog JSON logging. NO content (glosses/text/audio) is logged by default (MASTER_PROMPT §6)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

REDACT_KEYS = {
    "password",
    "token",
    "refresh_token",
    "access_token",
    "text",
    "glosses",
    "corrected_text",
    "corrected_glosses",
    "audio",
    "api_key",
    "code",
    "otp",
    "secret",
}


def _redact(_: Any, __: str, event: dict[str, Any]) -> dict[str, Any]:
    for k in list(event):
        if k in REDACT_KEYS:
            event[k] = "[redacted]"
    return event


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
