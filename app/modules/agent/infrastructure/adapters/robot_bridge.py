"""Forward agent actions to robot sockets. Motion commands always require explicit user confirmation and the
robot vendor's safety layer has final authority (MASTER_PROMPT §6)."""

from __future__ import annotations

from typing import Any

MOTION_ACTIONS = frozenset({"robot.navigate"})


def to_device_action(action: dict[str, Any], confirmed: bool) -> dict[str, Any] | None:
    if action.get("type") in MOTION_ACTIONS and not confirmed:
        return None
    return {"action": action["type"], "params": {k: v for k, v in action.items() if k != "type"}}
