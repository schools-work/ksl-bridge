"""Per-session sign state machine: IDLE → SIGNING → FINALISING → IDLE."""

from __future__ import annotations

from enum import Enum


class SignState(str, Enum):
    IDLE = "idle"
    SIGNING = "signing"
    FINALISING = "finalising"


_ALLOWED = {
    SignState.IDLE: {SignState.SIGNING},
    SignState.SIGNING: {SignState.FINALISING, SignState.IDLE},
    SignState.FINALISING: {SignState.IDLE, SignState.SIGNING},
}


class SignStateMachine:
    def __init__(self) -> None:
        self.state = SignState.IDLE
        self.revision = 0

    def to(self, new: SignState) -> None:
        if new not in _ALLOWED[self.state]:
            raise ValueError(f"illegal transition {self.state} → {new}")
        if new is SignState.SIGNING and self.state is not SignState.SIGNING:
            self.revision = 0
        self.state = new

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision
