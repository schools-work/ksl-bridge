import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ksl_core.normalize import FEATURE_DIM, StreamingNormalizer, build_features, interp_gaps, normalize_sequence


def _body(T: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.3, 0.7, size=(T, 107, 3)).astype(np.float32)
    x[:, 42 + 11] = [0.4, 0.5, 0.0]  # shoulders: a steady torso
    x[:, 42 + 12] = [0.6, 0.5, 0.0]
    return x


@settings(max_examples=40, deadline=None)
@given(dx=st.floats(-0.2, 0.2), dy=st.floats(-0.2, 0.2), s=st.floats(0.5, 2.0))
def test_translation_scale_invariance(dx: float, dy: float, s: float) -> None:
    x = _body()
    y = x.copy()
    y[..., 0] = (y[..., 0] + dx) * s
    y[..., 1] = (y[..., 1] + dy) * s
    y[..., 2] *= s
    assert np.allclose(normalize_sequence(x), normalize_sequence(y), atol=1e-4)


@settings(max_examples=30, deadline=None)
@given(theta=st.floats(-0.6, 0.6))
def test_roll_invariance(theta: float) -> None:
    x = _body()
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], np.float32)
    y = x.copy()
    y[..., :2] = (x[..., :2] - 0.5) @ R.T + 0.5
    assert np.allclose(normalize_sequence(x), normalize_sequence(y), atol=1e-4)


def test_features_shape_and_mask() -> None:
    x = _body()
    x[3:5, 0:21] = np.nan
    f, m = build_features(normalize_sequence(x, max_gap=0))
    assert f.shape == (20, FEATURE_DIM) and np.isfinite(f).all()
    assert not m[3].all()


def test_interp_fills_only_short_bounded_gaps() -> None:
    x = np.zeros((12, 1, 3), np.float32)
    x[:, 0, 0] = np.arange(12)
    x[3:5] = np.nan  # short, bounded → filled
    x[9:] = np.nan  # trailing → not filled
    y = interp_gaps(x, max_gap=3)
    assert np.allclose(y[3:5, 0, 0], [3, 4])
    assert np.isnan(y[9:]).all()


def test_streaming_converges_to_batch() -> None:
    x = _body(T=60)
    s = StreamingNormalizer()
    online = s.update(x)
    assert np.allclose(online[-1], normalize_sequence(x)[-1], atol=1e-3)
