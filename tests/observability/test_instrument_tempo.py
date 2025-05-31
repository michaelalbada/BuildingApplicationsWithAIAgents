import pytest
import importlib
import sys

@pytest.fixture(autouse=True)
def setup_instrument_tempo(monkeypatch):
    """
    Monkeypatch OTLPSpanExporter to a dummy that records spans,
    and replace BatchSpanProcessor with a SimpleSpanProcessor so spans
    are exported immediately.
    """
    exported_spans = []

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

    # Replace BatchSpanProcessor with a factory returning SimpleSpanProcessor
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    def fake_batch_processor(exporter):
        return SimpleSpanProcessor(exporter)
    monkeypatch.setattr(
        "opentelemetry.sdk.trace.export.BatchSpanProcessor",
        fake_batch_processor
    )

    # Monkeypatch RequestsInstrumentor.instrument to no-op
    monkeypatch.setattr(
        "opentelemetry.instrumentation.requests.RequestsInstrumentor.instrument",
        lambda self: None
    )

    # Re-import instrument_tempo so patches apply
    if "common.observability.instrument_tempo" in sys.modules:
        del sys.modules["common.observability.instrument_tempo"]
    instrument_tempo = importlib.import_module("common.observability.instrument_tempo")

    yield instrument_tempo, exported_spans


def test_do_work_generates_spans(setup_instrument_tempo):
    instrument_tempo, exported_spans = setup_instrument_tempo

    instrument_tempo.do_work()

    # Now SimpleSpanProcessor should have exported spans immediately
    assert len(exported_spans) >= 4

    span_names = [span.name for span in exported_spans]
    assert "parent-span" in span_names
    assert "child-span-0" in span_names
    assert "child-span-1" in span_names
    assert "child-span-2" in span_names
