"""Ports the pipeline depends on (implemented by realtime and infrastructure)."""

from __future__ import annotations

from typing import Any, Protocol


class Broadcaster(Protocol):
    async def publish(
        self,
        room: str,
        msg: dict[str, Any],
        binary: bytes | None = None,
        exclude_member: str | None = None,
        only_roles: list[str] | None = None,
        only_member: str | None = None,
        only_modality_out: list[str] | None = None,
    ) -> None: ...


class JobClient(Protocol):
    async def call(self, task: str, payload: dict[str, Any], deadline_ms: int) -> dict[str, Any] | None:
        """Submit a job to a worker stream; returns the output or None on timeout/no worker."""
        ...
