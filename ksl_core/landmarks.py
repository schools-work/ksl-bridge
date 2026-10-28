"""KSLB binary landmark codec (contracts/landmarks/landmark_spec_v1.json; binary_frames_v1.md)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ._generated.landmark_spec import (
    FACE_IDX,
    HEADER_STRUCT,
    MAGIC,
    N_POINTS,
    OFFSET_FACE,
    OFFSET_LEFT_HAND,
    OFFSET_POSE,
    OFFSET_RIGHT_HAND,
    POSE_NAMED,
    SPEC_VERSION,
)
from .errors import InvalidFrame, SpecVersionMismatch

SL_LH = slice(OFFSET_LEFT_HAND, OFFSET_LEFT_HAND + 21)
SL_RH = slice(OFFSET_RIGHT_HAND, OFFSET_RIGHT_HAND + 21)
SL_POSE = slice(OFFSET_POSE, OFFSET_POSE + 25)
SL_FACE = slice(OFFSET_FACE, OFFSET_FACE + 40)

POSE_NOSE = POSE_NAMED["nose"]
POSE_L_SHOULDER = POSE_NAMED["left_shoulder"]
POSE_R_SHOULDER = POSE_NAMED["right_shoulder"]
POSE_L_HIP = POSE_NAMED["left_hip"]
POSE_R_HIP = POSE_NAMED["right_hip"]

_HDR = struct.Struct(HEADER_STRUCT)
MAX_FRAMES_PER_BATCH = 64

__all__ = [
    "FACE_IDX",
    "N_POINTS",
    "SL_FACE",
    "SL_LH",
    "SL_POSE",
    "SL_RH",
    "SPEC_VERSION",
    "LandmarkBatch",
    "pack",
    "unpack",
]


@dataclass(slots=True)
class LandmarkBatch:
    """T frames. xyz: (T, 107, 3) float32 with NaN where missing; mask: (T, 107) bool (True = present)."""

    seq_start: int
    t0_ms: int
    fps: float
    xyz: np.ndarray
    mask: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.xyz.shape[0])


def pack(batch: LandmarkBatch) -> bytes:
    t = batch.xyz.shape[0]
    hdr = _HDR.pack(MAGIC, SPEC_VERSION, 0, t, batch.seq_start, batch.t0_ms, batch.fps)
    coords: bytes = np.nan_to_num(batch.xyz, nan=0.0).astype("<f2").tobytes()
    bits: bytes = np.packbits(batch.mask.reshape(-1).astype(np.uint8)).tobytes()
    return hdr + coords + bits


def unpack(buf: bytes, max_frames: int = MAX_FRAMES_PER_BATCH) -> LandmarkBatch:
    """Validate and decode. Raises InvalidFrame before touching untrusted sizes."""
    if len(buf) < _HDR.size:
        raise InvalidFrame("frame shorter than header")
    magic, ver, _flags, t, seq, t0, fps = _HDR.unpack_from(buf, 0)
    if magic != MAGIC:
        raise InvalidFrame("bad magic")
    if ver != SPEC_VERSION:
        raise SpecVersionMismatch(f"spec version {ver} != {SPEC_VERSION}")
    if not (0 < t <= max_frames):
        raise InvalidFrame(f"n_frames {t} out of range")
    if not (0 < fps <= 120):
        raise InvalidFrame("fps out of range")
    n = t * N_POINTS * 3
    nbits = t * N_POINTS
    expected = _HDR.size + n * 2 + (nbits + 7) // 8
    if len(buf) != expected:
        raise InvalidFrame(f"size {len(buf)} != expected {expected}")
    off = _HDR.size
    coords = np.frombuffer(buf, dtype="<f2", count=n, offset=off).astype(np.float32)
    if not np.isfinite(coords).all():
        raise InvalidFrame("non-finite coordinates")
    off += n * 2
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8, offset=off, count=(nbits + 7) // 8))[:nbits]
    mask = bits.astype(bool).reshape(t, N_POINTS)
    xyz = coords.reshape(t, N_POINTS, 3).copy()
    xyz[~mask] = np.nan
    return LandmarkBatch(seq, t0, float(fps), xyz, mask)
