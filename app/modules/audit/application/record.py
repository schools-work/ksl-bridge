"""Public entry point for recording audit events (delegates to the platform helper)."""

from app.platform.audit import record

__all__ = ["record"]
