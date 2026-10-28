import json
from pathlib import Path

import numpy as np

from ml.decoding.ctc_beam import CTCPrefixBeamSearch


def test_decoder_matches_golden() -> None:
    for case in json.loads(Path(__file__).with_name("cases.json").read_text()):
        d = CTCPrefixBeamSearch(case["itos"], beam=case["beam"], beta=case["beta"])
        d.step(np.array(case["log_probs"]))
        got = [[g, round(s, 6)] for g, s in d.nbest(3)]
        assert [g for g, _ in got] == [g for g, _ in case["expected"]], case["name"]
        assert np.allclose([s for _, s in got], [s for _, s in case["expected"]], atol=1e-5), case["name"]
