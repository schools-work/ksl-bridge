"""CodeCarbon energy/emissions tracking for every training run (Sustainable AI; ml.v_training_footprint)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def tracked(fn: Callable[..., T], project: str = "ksl-bridge") -> Callable[..., tuple[T, dict[str, float]]]:
    def wrap(*a: Any, **kw: Any) -> tuple[T, dict[str, float]]:
        try:
            from codecarbon import EmissionsTracker

            tr = EmissionsTracker(project_name=project, log_level="error", save_to_file=False)
        except ImportError:
            tr = None
        if tr:
            tr.start()
        try:
            result = fn(*a, **kw)
        finally:
            kg = tr.stop() if tr else None
        energy = float(getattr(getattr(tr, "final_emissions_data", None), "energy_consumed", 0.0) or 0.0) if tr else 0.0
        return result, {"emissions_kgco2e": float(kg or 0.0), "energy_kwh": energy}

    return wrap
