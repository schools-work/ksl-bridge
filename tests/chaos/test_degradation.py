"""With NO workers running, rooms must still relay text (graceful degradation, MASTER_PROMPT §4)."""

import uuid

import pytest

pytestmark = pytest.mark.integration


def test_text_relay_and_kb_without_workers(client) -> None:  # type: ignore[no-untyped-def]
    email = f"c{uuid.uuid4().hex[:6]}@test.ke"
    client.post("/auth/register", json={"display_name": "C", "email": email, "password": "Strong-pass-123"})
    tok = client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    venue = "00000000-0000-4000-8000-00000000a001"
    ans = client.post(f"/venues/{venue}/kb/ask", headers=h, json={"question": "Where is the toilet?"}).json()
    assert ans["answer"] and "toilet" in ans["answer"].lower()
    none = client.post(f"/venues/{venue}/kb/ask", headers=h, json={"question": "Do you sell televisions?"}).json()
    assert none["answer"] is None and none["escalate"] is True
    assert client.get("/readyz").json()["checks"]["db"] == "ok"
