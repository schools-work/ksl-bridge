"""Export UtteranceIR JSON Schema and the OpenAPI document into contracts/ (CI checks drift)."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    from ksl_core.ir import UtteranceIR

    (ROOT / "contracts/ir/utterance_ir_v1.json").write_text(
        json.dumps(UtteranceIR.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
    os.environ.setdefault("KSL_ENV", "test")
    os.environ.setdefault("KSL_METRICS_ENABLED", "false")
    from app.main import create_app

    (ROOT / "contracts/openapi/openapi_v1.json").write_text(
        json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    )
    print("exported contracts/ir/utterance_ir_v1.json and contracts/openapi/openapi_v1.json")


if __name__ == "__main__":
    main()
