import asyncio

from app.modules.realtime.connection import ClientConn


class _WS:
    async def send_text(self, _: str) -> None:
        await asyncio.sleep(0)

    async def send_bytes(self, _: bytes) -> None:
        await asyncio.sleep(0)


def test_partials_dropped_finals_kept() -> None:
    c = ClientConn(_WS(), "s", "user:x", maxsize=4)  # type: ignore[arg-type]
    for _ in range(4):
        assert c.enqueue({"type": "utterance.partial"})
    assert not c.enqueue({"type": "utterance.partial"})  # full → partial dropped
    assert c.enqueue({"type": "utterance.final"})  # final evicts partials
    types = [c.q.get_nowait()[0]["type"] for _ in range(c.q.qsize())]
    assert types[-1] == "utterance.final"


def test_envelope_sequence_and_replay() -> None:
    c = ClientConn(_WS(), "s", "user:x")  # type: ignore[arg-type]
    for i in range(3):
        c.envelope({"type": "ping", "data": {"t": i}})
    assert len(c.replay_after(1)) == 2
