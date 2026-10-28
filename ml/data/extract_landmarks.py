"""Video → (T,107,3) landmarks with local MediaPipe Holistic (the 'sensor'; ADR-0003)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ksl_core.landmarks import FACE_IDX, N_POINTS


def extract(video_path: str | Path, model_complexity: int = 1) -> tuple[np.ndarray, float]:
    import cv2  # type: ignore[import-not-found]
    import mediapipe as mp  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[np.ndarray] = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=model_complexity, refine_face_landmarks=False
    ) as holo:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            res = holo.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            out = np.full((N_POINTS, 3), np.nan, np.float32)

            def put(
                lms: object, offset: int, idx: list[int] | tuple[int, ...] | None = None, out: np.ndarray = out
            ) -> None:
                if lms is None:
                    return
                pts = lms.landmark  # type: ignore[attr-defined]
                sel = idx if idx is not None else range(len(pts))
                for k, j in enumerate(sel):
                    out[offset + k] = (pts[j].x, pts[j].y, pts[j].z)

            put(res.left_hand_landmarks, 0)
            put(res.right_hand_landmarks, 21)
            put(res.pose_landmarks, 42, list(range(25)))
            put(res.face_landmarks, 67, FACE_IDX)
            frames.append(out)
    cap.release()
    return (np.stack(frames) if frames else np.zeros((0, N_POINTS, 3), np.float32)), fps


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract KSL Bridge landmarks from videos")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in args.inputs:
        xyz, fps = extract(p)
        stem = Path(p).stem
        np.save(out / f"{stem}.npy", xyz.astype(np.float16))
        (out / f"{stem}.json").write_text(
            json.dumps({"source": str(p), "fps": fps, "n_frames": len(xyz), "spec_version": 1})
        )
        print(f"{p}: {len(xyz)} frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
