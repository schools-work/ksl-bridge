"""Load test: N rooms × 2 clients exchanging text over WebSockets. Run: locust -f tests/load/locustfile_ws.py"""

from __future__ import annotations

import json
import time
import uuid

from locust import User, between, events, task


class WsRoomUser(User):
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self) -> None:
        import httpx
        from websockets.sync.client import connect

        email = f"load{uuid.uuid4().hex[:8]}@test.ke"
        with httpx.Client(base_url=self.host) as c:
            c.post("/auth/register", json={"display_name": "L", "email": email, "password": "Strong-pass-123"})
            tok = c.post("/auth/login", json={"email": email, "password": "Strong-pass-123"}).json()["access_token"]
            self.room = c.post("/rooms", headers={"Authorization": f"Bearer {tok}"}, json={}).json()
        self.ws = connect(self.host.replace("http", "ws") + "/ws/v1")
        self.seq = 1
        self.ws.send(json.dumps({"v": 1, "type": "hello", "seq": 1, "ts": 0, "data": {"token": tok, "client": {}}}))
        self.ws.recv()
        self._send("room.join", room_id=self.room["room_id"], role="signer")

    def _send(self, t: str, **data: object) -> None:
        self.seq += 1
        self.ws.send(json.dumps({"v": 1, "type": t, "seq": self.seq, "ts": 0, "data": data}))

    @task
    def text(self) -> None:
        t0 = time.perf_counter()
        self._send("text.send", text="Habari, I want water")
        while True:
            m = json.loads(self.ws.recv())
            if m["type"] == "utterance.final":
                break
        events.request.fire(
            request_type="WS",
            name="text->final",
            response_time=(time.perf_counter() - t0) * 1000,
            response_length=0,
            exception=None,
        )

    def on_stop(self) -> None:
        self.ws.close()
