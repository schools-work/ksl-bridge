"""ModelAdapter interface + registry. Bootstrap and our own models are interchangeable (MASTER_PROMPT §4.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    task: str
    name: str
    version: str
    device: str
    is_bootstrap: bool = False


class ModelUnavailable(RuntimeError):
    """No deployed model for this task (or its artefact couldn't be loaded)."""


class ModelAdapter(ABC):
    info: ModelInfo

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pure: inputs → outputs, same length and order."""

    def warmup(self) -> None:  # noqa: B027 (optional hook; default does nothing)
        """Run dummy inputs so kernels compile before real traffic."""


REGISTRY: dict[str, type[ModelAdapter]] = {}


def register(name: str):  # type: ignore[no-untyped-def]
    def deco(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        REGISTRY[name] = cls
        return cls

    return deco
