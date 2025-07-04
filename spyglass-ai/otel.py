import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def _get_exporter():
    """Get the appropriate exporter based on environment configuration."""
    api_key = os.environ.get("SPYGLASS_API_KEY")
    
    if api_key:
        # Use OTLP exporter with API key as bearer token
        headers = {"Authorization": f"Bearer {api_key}"}
        # Default endpoint - can be overridden with OTEL_EXPORTER_OTLP_ENDPOINT
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "https://api.spyglass-ai.com/v1/traces")
        return OTLPSpanExporter(endpoint=endpoint, headers=headers)
    else:
        # Fall back to console exporter if no API key is provided
        return ConsoleSpanExporter()


provider = TracerProvider()
processor = BatchSpanProcessor(_get_exporter())
provider.add_span_processor(processor)

# Sets the global default tracer provider
trace.set_tracer_provider(provider)

# Creates a tracer from the global tracer provider
spyglass_tracer = trace.get_tracer("spyglass-tracer")