"""ksl_core: dependency-light code shared by the service, the workers and the ML code.

Only NumPy and Pydantic are allowed here. Anything that runs in both training and serving
(normalisation, features, vocabularies, the landmark codec, the IR) lives here, so training and
serving can't drift apart.
"""

__all__ = ["errors", "ir", "landmarks", "normalize", "smoothing", "text", "vocab"]
__version__ = "0.1.0"
