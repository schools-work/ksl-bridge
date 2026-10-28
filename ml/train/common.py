"""Shared training utilities: config, seeding, manifests, vocab building, device selection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["_path"] = str(path)
    return cfg


def config_sha(cfg: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True).encode()
    ).digest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()  # noqa: S607
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """JSONL: one sample per line {sample_id, signer_id, glosses, landmarks_path, split, dominant_hand, …}."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def by_split(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for it in items:
        out[it["split"]].append(it)
    return out
