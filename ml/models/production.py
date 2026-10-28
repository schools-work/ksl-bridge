"""Sign production: dictionary medoid clips + Hermite transitions + non-manual overlay (Guide Part IV §11)."""

from __future__ import annotations

import numpy as np


def hermite_blend(a_end: np.ndarray, a_vel: np.ndarray, b_start: np.ndarray, b_vel: np.ndarray, k: int) -> np.ndarray:
    t = np.linspace(0, 1, k + 2)[1:-1][:, None, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * a_end + h10 * a_vel * k + h01 * b_start + h11 * b_vel * k


def trim_rest(clip: np.ndarray, thr: float = 0.02) -> np.ndarray:
    if len(clip) < 3:
        return clip
    v = np.nan_to_num(np.linalg.norm(np.diff(clip[:, :42], axis=0), axis=-1)).mean(-1)
    active = np.where(v > thr)[0]
    if len(active) == 0:
        return clip
    return clip[max(0, active[0] - 2) : active[-1] + 3]


def produce(
    glosses: list[str],
    dictionary: dict[str, np.ndarray],
    fs_letters: dict[str, np.ndarray],
    k: int = 6,
    speed: float = 1.0,
    question: str | None = None,
    face_templates: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    clips: list[np.ndarray] = []
    for g in glosses:
        if g.startswith("FS:"):
            for ch in g[3:]:
                if ch in fs_letters:
                    clips.append(np.repeat(fs_letters[ch][-1:], 6, axis=0))
        elif g in dictionary:
            clips.append(trim_rest(dictionary[g]))
    if not clips:
        return np.zeros((0, 107, 3), np.float32)
    out = [clips[0]]
    for nxt in clips[1:]:
        prev = out[-1]
        a_vel = prev[-1] - prev[-2] if len(prev) > 1 else np.zeros_like(prev[-1])
        b_vel = nxt[1] - nxt[0] if len(nxt) > 1 else np.zeros_like(nxt[0])
        out.append(hermite_blend(prev[-1], a_vel, nxt[0], b_vel, k))
        out.append(nxt)
    seq = np.concatenate(out, axis=0)
    if question and face_templates and question in face_templates:
        tpl = face_templates[question]
        n = min(len(seq), len(clips[-1]))
        w = np.linspace(0, 1, n)[:, None, None]
        seq[-n:, 67:107] = (1 - w) * seq[-n:, 67:107] + w * (seq[-n:, 67:107] + tpl)
    if speed != 1.0:
        T = len(seq)
        newT = max(2, int(T / speed))
        src = np.linspace(0, T - 1, newT)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (src - i0)[:, None, None]
        seq = (1 - w) * seq[i0] + w * seq[i1]
    return seq.astype(np.float32)
