import uuid

import pytest

pytestmark = pytest.mark.integration


def test_user_cannot_read_others(client) -> None:  # type: ignore[no-untyped-def]
    tokens = []
    for _ in range(2):
        email = f"x{uuid.uuid4().hex[:6]}@test.ke"
        client.post("/auth/register", json={"display_name": "X", "email": email, "password": "Strong-pass-123"})
        tokens.append(client.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json())
    me0 = client.get("/me", headers={"Authorization": f"Bearer {tokens[0]['access_token']}"}).json()
    me1 = client.get("/me", headers={"Authorization": f"Bearer {tokens[1]['access_token']}"}).json()
    assert me0["id"] != me1["id"]
    export = client.get("/me/export", headers={"Authorization": f"Bearer {tokens[0]['access_token']}"}).json()
    assert len(export["user"]) == 1 and export["user"][0]["id"] == me0["id"]
