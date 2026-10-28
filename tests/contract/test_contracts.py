"""Contracts are the SSOT shared with Kotlin: generated code must be in sync."""

import json
import subprocess
import sys
from pathlib import Path

from ksl_core._generated import landmark_spec as spec
from ksl_core._generated.ws_messages import C2S_TYPES, MESSAGE_TYPES, ErrorCode
from ksl_core.landmarks import FACE_IDX, N_POINTS
from ksl_core.normalize import FEATURE_DIM

ROOT = Path(__file__).resolve().parents[2]


def test_generated_code_has_no_drift() -> None:
    r = subprocess.run(
        [sys.executable, "contracts/codegen/gen_python.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_landmark_spec_consistent() -> None:
    j = json.loads((ROOT / "contracts/landmarks/landmark_spec_v1.json").read_text())
    assert N_POINTS == j["n_points"] == sum(g["count"] for g in j["groups"])
    assert FACE_IDX == tuple(j["groups"][3]["indices"]) and len(FACE_IDX) == 40
    assert FEATURE_DIM == j["features"]["dim"] == spec.FEATURE_DIM


def test_protocol_and_error_codes() -> None:
    assert {"hello", "text.send", "feedback"} <= C2S_TYPES <= MESSAGE_TYPES
    assert ErrorCode.AUTH_REQUIRED == 4001 and ErrorCode.PIPELINE_UNAVAILABLE == 4500


def test_utterance_ir_schema_exported() -> None:
    from ksl_core.ir import UtteranceIR

    exported = json.loads((ROOT / "contracts/ir/utterance_ir_v1.json").read_text())
    assert exported == json.loads(json.dumps(UtteranceIR.model_json_schema(), sort_keys=True))


def test_kotlin_codegen_runs(tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, "contracts/codegen/gen_kotlin.py", "--out", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    kt = next(tmp_path.rglob("LandmarkSpec.kt")).read_text()
    assert "N_POINTS: Int = 107" in kt
