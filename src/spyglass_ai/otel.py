import os
import grpc
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class SpyglassOtelError(Exception):
    """Base exception for Spyglass OpenTelemetry configuration errors."""
    pass

class ExporterConfigurationError(SpyglassOtelError):
    """Raised when exporter configuration is invalid."""
    pass

def _create_exporter():
    """Create and return an OTLP gRPC span exporter based on environment variables."""
    api_key = os.getenv("SPYGLASS_API_KEY")
    
    kwargs = {}
    kwargs["insecure"] = False
    kwargs["endpoint"] = "ingest.spyglass-ai.com"
    kwargs["compression"] = grpc.Compression.Gzip
    
    if not api_key:
        raise ExporterConfigurationError("SPYGLASS_API_KEY is not set")
    
    # Set Authorization header with Bearer token
    kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    
    return OTLPSpanExporter(**kwargs)

# Create the tracer provider and exporter
provider = TracerProvider()
exporter = _create_exporter()
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)

# Sets the global default tracer provider
trace.set_tracer_provider(provider)

# Creates a tracer from the global tracer provider
spyglass_tracer = trace.get_tracer("spyglass-tracer")