"""Write experiments, runs, metrics, model versions and evaluations to the ml schema (role ksl_trainer)."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from ml.db import engine

from .common import config_sha, git_commit


def _x(sql: str, **params: Any) -> Any:
    import sqlalchemy as sa

    with engine().begin() as c:
        return c.execute(sa.text(sql), params)


def ensure_experiment(task: str, name: str, hypothesis: str, rq: str | None = None) -> uuid.UUID:
    row = _x(
        "INSERT INTO ml.experiments (task, name, hypothesis, research_question) VALUES "
        "(CAST(:t AS shared.model_task), :n, :h, :rq) ON CONFLICT (name) DO UPDATE SET hypothesis = EXCLUDED.hypothesis "
        "RETURNING id",
        t=task,
        n=name,
        h=hypothesis,
        rq=rq,
    ).first()
    return uuid.UUID(str(row[0]))


def start_run(
    experiment_id: uuid.UUID,
    task: str,
    cfg: dict[str, Any],
    seed: int,
    dataset_build_id: str | None = None,
    parent_run_id: str | None = None,
) -> uuid.UUID:
    row = _x(
        "INSERT INTO ml.training_runs (experiment_id, task, parent_run_id, dataset_build_id, config, config_sha256, "
        "code_commit, seed, status, started_at) VALUES (:e, CAST(:t AS shared.model_task), :p, :b, CAST(:c AS jsonb), "
        ":h, :g, :s, 'running', now()) RETURNING id",
        e=experiment_id,
        t=task,
        p=parent_run_id,
        b=dataset_build_id,
        c=json.dumps(cfg, default=str),
        h=config_sha(cfg),
        g=git_commit(),
        s=seed,
    ).first()
    return uuid.UUID(str(row[0]))


def log_metric(run_id: uuid.UUID, key: str, value: float, step: int) -> None:
    _x(
        "INSERT INTO ml.run_metrics (run_id, key, step, value) VALUES (:r, :k, :s, :v) "
        "ON CONFLICT (run_id, key, step) DO UPDATE SET value = EXCLUDED.value",
        r=run_id,
        k=key,
        s=step,
        v=value,
    )


def finish_run(
    run_id: uuid.UUID,
    ok: bool,
    footprint: dict[str, float] | None = None,
    error: str | None = None,
    artifact_uri: str | None = None,
) -> None:
    fp = footprint or {}
    _x(
        "UPDATE ml.training_runs SET status = CAST(:st AS shared.job_status), finished_at = now(), error = :e, "
        "energy_kwh = :kwh, emissions_kgco2e = :kg, artifact_uri = COALESCE(:a, artifact_uri) WHERE id = :r",
        st="succeeded" if ok else "failed",
        e=error,
        kwh=fp.get("energy_kwh"),
        kg=fp.get("emissions_kgco2e"),
        a=artifact_uri,
        r=run_id,
    )


def register_model(
    task: str,
    name: str,
    version: str,
    run_id: uuid.UUID | None,
    artifact: Path,
    fmt: str,
    artifact_uri: str | None = None,
    params_count: int | None = None,
    vocabulary_id: str | None = None,
    is_bootstrap: bool = False,
    bootstrap_source: str | None = None,
    quantization: str | None = None,
) -> uuid.UUID:
    sha = hashlib.sha256(artifact.read_bytes()).digest()
    row = _x(
        "INSERT INTO ml.model_versions (task, name, version, run_id, is_bootstrap, bootstrap_source, vocabulary_id, "
        "format, artifact_uri, sha256, size_bytes, params_count, quantization) VALUES (CAST(:t AS shared.model_task), "
        ":n, :v, :r, :b, :bs, :voc, :f, :u, :h, :sz, :pc, :q) RETURNING id",
        t=task,
        n=name,
        v=version,
        r=run_id,
        b=is_bootstrap,
        bs=bootstrap_source,
        voc=vocabulary_id,
        f=fmt,
        u=artifact_uri or str(artifact),
        h=sha,
        sz=artifact.stat().st_size,
        pc=params_count,
        q=quantization,
    ).first()
    return uuid.UUID(str(row[0]))


def record_evaluation(
    model_id: uuid.UUID,
    build_id: str,
    split: str,
    suite: str,
    results: list[dict[str, Any]],
    perturbation: str | None = None,
    decoder_config_id: str | None = None,
) -> uuid.UUID:
    row = _x(
        "INSERT INTO ml.evaluations (model_version_id, decoder_config_id, dataset_build_id, split, suite, perturbation, "
        "code_commit) VALUES (:m, :d, :b, CAST(:s AS shared.data_split), :su, :p, :g) RETURNING id",
        m=model_id,
        d=decoder_config_id,
        b=build_id,
        s=split,
        su=suite,
        p=perturbation,
        g=git_commit(),
    ).first()
    eid = uuid.UUID(str(row[0]))
    for r in results:
        _x(
            "INSERT INTO ml.evaluation_results (evaluation_id, metric, slice_key, slice_value, value, ci_low, ci_high, n) "
            "VALUES (:e, :m, :k, :v, :x, :lo, :hi, :n)",
            e=eid,
            m=r["metric"],
            k=r.get("slice_key", "all"),
            v=r.get("slice_value", "all"),
            x=r["value"],
            lo=r.get("ci_low"),
            hi=r.get("ci_high"),
            n=r["n"],
        )
    return eid


def register_vocabulary(name: str, version: str, kind: str, itos: list[str]) -> uuid.UUID:
    sha = hashlib.sha256("\n".join(itos).encode()).digest()
    row = _x(
        "INSERT INTO ml.vocabularies (name, version, kind, size, sha256) VALUES (:n, :v, :k, :s, :h) "
        "ON CONFLICT (name, version) DO UPDATE SET size = EXCLUDED.size RETURNING id",
        n=name,
        v=version,
        k=kind,
        s=len(itos),
        h=sha,
    ).first()
    vid = uuid.UUID(str(row[0]))
    for i, tok in enumerate(itos):
        _x(
            "INSERT INTO ml.vocabulary_tokens (vocabulary_id, idx, token) VALUES (:v, :i, :t) ON CONFLICT DO NOTHING",
            v=vid,
            i=i,
            t=tok,
        )
    return vid
