"""Build the C++ decoder and compare it with the Python reference on the golden cases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.decoding.ctc_beam import CTCPrefixBeamSearch


def build():  # type: ignore[no-untyped-def]
    from torch.utils.cpp_extension import load

    return load(name="ksl_ctc_beam", sources=[str(Path(__file__).with_name("ctc_beam.cpp"))], verbose=False)


def main() -> None:
    ext = build()
    cases = json.loads(Path("tests/golden/cslr/cases.json").read_text())
    for case in cases:
        lp = np.array(case["log_probs"], dtype=np.float32)
        ref = CTCPrefixBeamSearch(case["itos"], beam=case["beam"], beta=case["beta"])
        ref.step(lp)
        cpp = ext.beam_search(torch.from_numpy(lp), case["beam"], case["beta"], -8.0, [1, 3, 4])
        ok = [case["itos"][i] for i in cpp[0][0]] == ref.nbest(1)[0][0]
        print(case["name"], "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
