import numpy as np
import pytest

from ksl_core.errors import InvalidFrame, SpecVersionMismatch
from ksl_core.landmarks import LandmarkBatch, pack, unpack


def _batch(seed: int = 1) -> LandmarkBatch:
    rng = np.random.default_rng(seed)
    xyz = rng.random((4, 107, 3)).astype(np.float32)
    mask = rng.random((4, 107)) > 0.2
    xyz[~mask] = np.nan
    return LandmarkBatch(10, 1234, 30.0, xyz, mask)


def test_roundtrip() -> None:
    b0 = _batch()
    b = unpack(pack(b0))
    assert (b.seq_start, b.t0_ms, b.n_frames) == (10, 1234, 4)
    assert (b.mask == b0.mask).all()
    assert np.allclose(b.xyz[b0.mask], b0.xyz[b0.mask], atol=1e-3)
    assert np.isnan(b.xyz[~b0.mask]).all()


def test_rejects_bad_magic_size_and_version() -> None:
    raw = bytearray(pack(_batch()))
    with pytest.raises(InvalidFrame):
        unpack(bytes(raw[:-1]))
    bad = bytearray(raw)
    bad[0:4] = b"XXXX"
    with pytest.raises(InvalidFrame):
        unpack(bytes(bad))
    bad = bytearray(raw)
    bad[4] = 9
    with pytest.raises(SpecVersionMismatch):
        unpack(bytes(bad))


def test_rejects_too_many_frames() -> None:
    rng = np.random.default_rng(0)
    xyz = rng.random((100, 107, 3)).astype(np.float32)
    with pytest.raises(InvalidFrame):
        unpack(pack(LandmarkBatch(0, 0, 30.0, xyz, np.ones((100, 107), bool))))
