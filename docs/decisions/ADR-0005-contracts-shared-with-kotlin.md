# ADR-0005: Language-neutral contracts with code generation

* **Status:** accepted
* **Date:** 2026-10-27

## Decision
`contracts/` holds JSON (Schema) definitions for the landmark layout, the WebSocket protocol, error codes and `UtteranceIR`, plus the generated OpenAPI. `contracts/codegen/gen_python.py` generates `ksl_core/_generated/*`, and `gen_kotlin.py` generates Kotlin data classes for the app repository. CI fails on drift (`gen_python.py --check`, `tests/contract/`). Any breaking change bumps the version (`*_v2`), and the old version stays supported until the app ships.
