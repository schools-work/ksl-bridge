"""Store a final utterance ONLY when the room stores transcripts (consented)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncConnection

from ksl_core.ir import UtteranceIR

from ..infrastructure.repositories import SqlRoomRepo


async def execute(conn: AsyncConnection, ir: UtteranceIR, member_id: str | None, force_link: bool = False) -> bool:
    """force_link=True stores a minimal row (no content) so clarifications/feedback can reference it."""
    repo = SqlRoomRepo(conn)
    stores = await repo.stores_transcript(ir.room_id)
    if not stores and not force_link:
        return False
    await repo.insert_utterance(ir, member_id, with_content=stores)
    return True
