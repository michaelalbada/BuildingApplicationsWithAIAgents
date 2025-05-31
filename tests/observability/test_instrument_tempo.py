import pytest
import importlib
import sys
import os

@pytest.fixture(autouse=True)
def setup_instrument_tempo(monkeypatch):
    """
    Monkeypatch OTLPSpanExporter to a dummy that records spans,
    and replace BatchSpanProcessor with SimpleSpanProcessor so spans
    are exported immediately.
    Also ensure 'src' is on sys.path so that common.observability.instrument_tempo can be imported.
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # Insert the 'src' directory (where common/observability/instrument_tempo.py lives)
    # into sys.path, so Python can find and import the module properly.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src_path = os.path.join(repo_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    # ─────────────────────────────────────────────────────────────────────────────

    exported_spans = []

    # DummyExporter: captures any spans passed to export()
    class DummyExporter:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, spans):
            exported_spans.extend(spans)

        def shutdown(self):
            pass

    # Monkeypatch OTLPSpanExporter to our DummyExporter
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter",
        DummyExporter
    )

    # Replace BatchSpanProcessor with SimpleSpanProcessor (so spans export immediately)
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    def fake_batch_processor(exporter):
        return SimpleSpanProcessor(exporter)

    monkeypatch.setattr(
        "opentelemetry.sdk.trace.export.BatchSpanProcessor",
        fake_batch_processor
    )

    # Disable the RequestsInstrumentor (no external instrumentation)
    monkeypatch.setattr(
        "opentelemetry.instrumentation.requests.RequestsInstrumentor.instrument",
        lambda self: None
    )

    # Re-import the module (clearing any previously cached version)
    module_name = "common.observability.instrument_tempo"
    if module_name in sys.modules:
        del sys.modules[module_name]
    instrument_tempo = importlib.import_module(module_name)

    # ─────────────────────────────────────────────────────────────────────────────
    # Override the tracer provider and tracer in the instrument_tempo module
    # so that do_work() uses DummyExporter + SimpleSpanProcessor.
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # 1. Create a new TracerProvider
    new_provider = TracerProvider()
    # 2. Attach a SimpleSpanProcessor that uses DummyExporter
    new_provider.add_span_processor(SimpleSpanProcessor(DummyExporter()))
    # 3. Override the global tracer provide
