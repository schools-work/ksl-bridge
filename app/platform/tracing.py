"""OpenTelemetry tracing: one trace per utterance, a span per pipeline stage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

_tracer = trace.get_tracer("ksl-bridge")


def configure_tracing(endpoint: str) -> None:
    """Install an OTLP exporter when an endpoint is configured; otherwise spans are no-ops."""
    if not endpoint:
        return
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # exporter extra not installed
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "ksl-bridge-gateway"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
