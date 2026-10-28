# C++/CUDA CTC prefix beam decoder (only if profiling demands it)

The Python decoder (`ml/decoding/ctc_beam.py`) is the reference. Port it here only when `torch.profiler` shows decoding above about 20% of chunk latency at the production beam size.

* `ctc_beam.cpp`: CPU reference port (same algorithm, the same `lm_cache`, the same `skip_ids`)
* Build: `python -m ml.kernels.ctc_beam_cuda.build` (torch `cpp_extension.load`)
* Acceptance: identical N-best to the Python decoder on `tests/golden/cslr/cases.json`, plus a measured speed-up
