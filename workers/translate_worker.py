"""CPU worker for gloss→text (g2t), text→gloss (t2g) and NLU. Model if deployed, else rule baselines."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from ml.models.nlu import KeywordIntentBaseline
from ml.text2gloss.rules import rule_gloss2text, rule_text2gloss
from ml.text2gloss.validator import GlossValidator

from .adapters import ModelAdapter, ModelInfo, register
from .loader import fetch_artifact, live_model
from .runner import StreamWorker


def _dictionary() -> tuple[set[str], dict[str, str], set[str]]:
    """Gloss dictionary, English→gloss synonyms and critical glosses from the lexicon schema."""
    import sqlalchemy as sa

    from .loader import _connect

    with _connect().connect() as c:
        glosses = {
            r[0]: bool(r[1])
            for r in c.execute(sa.text("SELECT gloss, is_critical FROM lexicon.glosses WHERE status <> 'rejected'"))
        }
        syn = {
            r[0].upper(): r[1]
            for r in c.execute(sa.text("SELECT phrase, gloss FROM lexicon.translations WHERE language = 'en'"))
        }
    return set(glosses), syn, {g for g, crit in glosses.items() if crit}


@register("g2t")
class Gloss2TextAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("gloss2text", "rules", "rules-1", "cpu", True)
        self.model: Any = None

    def load(self) -> None:
        m = live_model("gloss2text")
        if m is not None:
            from ml.models.seq2seq import Seq2SeqTranslator

            self.model = Seq2SeqTranslator.from_checkpoint(fetch_artifact(m))
            self.info = ModelInfo("gloss2text", m.name, m.version, "cpu", m.is_bootstrap)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        outs = []
        for i in inputs:
            text = self.model.translate(i["glosses"]) if self.model else rule_gloss2text(i["glosses"])
            outs.append({"text": text, "model_versions": {"g2t": self.info.version}})
        return outs


@register("t2g")
class Text2GlossAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("text2gloss", "rules", "rules-1", "cpu", True)
        self.validator: GlossValidator | None = None

    def load(self) -> None:
        try:
            d, syn, crit = _dictionary()
        except Exception:
            d, syn, crit = set(), {}, set()
        self.validator = GlossValidator(d, syn, crit)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert self.validator is not None
        outs = []
        for i in inputs:
            glosses, question = rule_text2gloss(i["text"], self.validator.dictionary, self.validator.synonyms)
            v = self.validator.validate(glosses)
            outs.append(
                {
                    "glosses": v.glosses,
                    "question": question,
                    "low_confidence": v.fingerspelled > 0,
                    "critical": v.critical,
                    "model_versions": {"t2g": self.info.version},
                }
            )
        return outs


@register("nlu")
class NluAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.info = ModelInfo("nlu", "keyword-baseline", "rules-1", "cpu", True)
        self.model: Any = KeywordIntentBaseline()

    def load(self) -> None:
        m = live_model("nlu")
        if m is not None:
            from ml.models.nlu import load_sklearn_pipeline

            self.model = load_sklearn_pipeline(fetch_artifact(m))
            self.info = ModelInfo("nlu", m.name, m.version, "cpu", m.is_bootstrap)

    def infer_batch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.model.predict(i.get("text") or "", i.get("glosses") or []) for i in inputs]


async def _run_all() -> None:
    import redis.asyncio as aioredis

    r = aioredis.from_url(os.environ.get("KSL_REDIS_URL", "redis://localhost:6379/0"))
    await asyncio.gather(
        StreamWorker(r, "g2t", Gloss2TextAdapter()).run(),
        StreamWorker(r, "t2g", Text2GlossAdapter()).run(),
        StreamWorker(r, "nlu", NluAdapter()).run(),
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(_run_all()))
