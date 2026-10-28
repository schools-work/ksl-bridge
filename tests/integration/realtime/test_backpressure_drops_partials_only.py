import pytest

from app.modules.realtime.connection import ClientConn

pytestmark = pytest.mark.integration


class _WS:
    async def send_text(self, _: str) -> None: ...
    async def send_bytes(self, _: bytes) -> None: ...


def test_final_never_dropped_under_pressure() -> None:
    c = ClientConn(_WS(), "s", "user:x", maxsize=8)  # type: ignore[arg-type]
    for _ in range(50):
        c.enqueue({"type": "utterance.partial"})
    for _ in range(3):
        assert c.enqueue({"type": "utterance.final"})
    kept = [c.q.get_nowait()[0]["type"] for _ in range(c.q.qsize())]
    assert kept.count("utterance.final") == 3
