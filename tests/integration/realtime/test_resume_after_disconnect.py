import json
import uuid

import pytest

pytestmark = pytest.mark.integration


def _msg(type_: str, seq: int, **data: object) -> str:
    return json.dumps({"v": 1, "type": type_, "seq": seq, "ts": 0, "data": data})


def test_resume_replays_missed_messages(client) -> None:  # type: ignore[no-untyped-def]
    email = f"z{uuid.uuid4().hex[:6]}@test.ke"
    client.post("/auth/register", json={"display_name": "Z", "email": email, "password": "Strong-pass-123"})
    tok = client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()["access_token"]
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_text(_msg("hello", 1, token=tok, client={}))
        welcome = json.loads(ws.receive_text())
        sid = welcome["data"]["session_id"]
        ws.send_text(_msg("ping", 2, t=7))
        assert json.loads(ws.receive_text())["type"] == "pong"
    with client.websocket_connect("/ws/v1") as ws:
        ws.send_text(_msg("hello", 1, token=tok, client={}, resume={"session_id": sid, "last_seq": 0}))
        first = json.loads(ws.receive_text())
        assert first["seq"] == 1 and first["type"] == "welcome"  # replayed from the previous session
