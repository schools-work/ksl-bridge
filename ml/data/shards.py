"""WebDataset shards (≈1 GB .tar) of normalised features for streaming training."""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


def write_shards(
    samples: Iterable[dict[str, Any]], out_dir: str | Path, prefix: str, max_bytes: int = 1 << 30
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    tar: tarfile.TarFile | None = None
    size = 0

    def _new() -> tarfile.TarFile:
        p = out / f"{prefix}-{len(paths):06d}.tar"
        paths.append(p)
        return tarfile.open(p, "w")

    for s in samples:
        if tar is None or size > max_bytes:
            if tar is not None:
                tar.close()
            tar, size = _new(), 0
        key = str(s["sample_id"])
        for name, payload in (("xyz.npy", _npy(s["xyz"])), ("meta.json", json.dumps(s["meta"]).encode())):
            info = tarfile.TarInfo(f"{key}.{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            size += len(payload)
    if tar is not None:
        tar.close()
    return paths


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float16))
    return buf.getvalue()


def read_shard(path: str | Path) -> Iterator[dict[str, Any]]:
    with tarfile.open(path) as tar:
        cur: dict[str, Any] = {}
        for m in tar.getmembers():
            key, _, name = m.name.partition(".")
            data = tar.extractfile(m).read()  # type: ignore[union-attr]
            if cur and cur["sample_id"] != key:
                yield cur
                cur = {}
            cur["sample_id"] = key
            if name == "xyz.npy":
                cur["xyz"] = np.load(io.BytesIO(data)).astype(np.float32)
            elif name == "meta.json":
                cur["meta"] = json.loads(data)
        if cur:
            yield cur
