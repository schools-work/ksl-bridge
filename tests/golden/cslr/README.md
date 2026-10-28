# Golden decoder tests

`cases.json` holds fixed frame posteriors and the N-best that `ml/decoding/ctc_beam.py` must reproduce. A change to the decoder that alters an output must update the golden file **in the same PR**, with the WER impact on the validation set stated.

Regenerate with `python -m tests.golden.cslr.regenerate` (reviewers check the diff).
