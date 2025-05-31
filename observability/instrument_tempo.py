# instrument_tempo.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

resource = Resource.create({"service.name": "my-python-service"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:3200",
    insecure=True,
)
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

RequestsInstrumentor().instrument()

tracer = trace.get_tracer(__name__)

def do_work():
    with tracer.start_as_current_span("parent-span") as parent:
        # Simulate work
        for i in range(3):
            with tracer.start_as_current_span(f"child-span-{i}"):
                print(f"Working on child {i}...")

if __name__ == "__main__":
    do_work()
    print("Finished sending spans to Tempo.")