"""Register a mobile artefact and (after gates) promote it with target='mobile' and a minimum app version."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.train.registry import register_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--task", default="islr")
    ap.add_argument("--name", default="islr_core_tflite")
    ap.add_argument("--version", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--uri", help="s3:// location after upload")
    a = ap.parse_args()
    mid = register_model(
        a.task, a.name, a.version, a.run_id, Path(a.artifact), "tflite", artifact_uri=a.uri, quantization="int8"
    )
    print(
        f"registered {mid}. Record gates, then: SELECT ml.promote_model('{mid}', 'mobile', '<you>', NULL, '<min app>');"
    )


if __name__ == "__main__":
    main()
