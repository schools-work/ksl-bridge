# contracts/ (language-neutral, shared with the Kotlin app)

| File | SSOT for | Generated into |
|---|---|---|
| `landmarks/landmark_spec_v1.json` | 107-point layout, face indices, slices, binary header | `ksl_core/_generated/landmark_spec.py`, Kotlin `LandmarkSpec.kt` |
| `realtime/ws_protocol_v1.json` | WebSocket envelope + every message (JSON Schema 2020-12) | `ksl_core/_generated/ws_messages.py` (type names + required fields), Kotlin sealed classes |
| `realtime/binary_frames_v1.md` | Binary frame kinds 0x01–0x04 | (documentation) |
| `realtime/error_codes_v1.json` | Close/error codes | Python + Kotlin enums |
| `ir/utterance_ir_v1.json` | `UtteranceIR` JSON Schema | **exported from** `ksl_core/ir.py` (`make contracts`) |
| `openapi/openapi_v1.json` | REST API | **exported from** the FastAPI app (`make contracts`) |

## Rules
* Breaking change → new version file (`*_v2.json`). Keep v1 working until the app ships v2.
* CI runs `python contracts/codegen/gen_python.py --check` and `tests/contract/`. Drift fails the build.
* Never hand-edit `ksl_core/_generated/`.
