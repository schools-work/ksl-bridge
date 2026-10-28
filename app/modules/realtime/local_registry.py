"""Sockets connected to THIS gateway replica, indexed by room."""

from __future__ import annotations

from collections import defaultdict

from .connection import ClientConn


class LocalRegistry:
    def __init__(self) -> None:
        self.by_room: dict[str, set[ClientConn]] = defaultdict(set)
        self.by_session: dict[str, ClientConn] = {}

    def add(self, room: str, c: ClientConn) -> None:
        self.by_room[room].add(c)
        self.by_session[c.session_id] = c

    def remove(self, room: str | None, c: ClientConn) -> None:
        self.by_session.pop(c.session_id, None)
        if room is None:
            return
        self.by_room[room].discard(c)
        if not self.by_room[room]:
            self.by_room.pop(room, None)

    def members(self, room: str) -> set[ClientConn]:
        return set(self.by_room.get(room, ()))

    def count(self) -> int:
        return len(self.by_session)
