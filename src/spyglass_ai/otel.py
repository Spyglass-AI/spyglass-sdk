import os
import grpc
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

class SpyglassOtelError(Exception):
    """Base exception for Spyglass OpenTelemetry configuration errors."""
    pass

class ExporterConfigurationError(SpyglassOtelError):
    """Raised when exporter configuration is invalid."""
    pass

class DeploymentConfigurationError(SpyglassOtelError):
    """Raised when deployment configuration is invalid."""
    pass

def _create_resource():
    """Create and return a Resource with deployment and service information."""
    resource_attributes = {}
    
    # Deployment information - this is required
    deployment_id = os.getenv("SPYGLASS_DEPLOYMENT_ID")
    if not deployment_id:
        raise DeploymentConfigurationError("SPYGLASS_DEPLOYMENT_ID is required but not set")
    
    # Use deployment_id for both service.name and deployment.id
    resource_attributes["service.name"] = deployment_id
    resource_attributes["deployment.id"] = deployment_id
    
    return Resource.create(resource_attributes)


def _create_exporter():
    """Create and return an OTLP gRPC span exporter based on environment variables."""
    api_key = os.getenv("SPYGLASS_API_KEY")
    
    # Check for custom endpoint (for development)
    endpoint = os.getenv("SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT", "https://ingest.spyglass-ai.com:443")
    
    kwargs = {}
    kwargs["endpoint"] = endpoint
    kwargs["compression"] = grpc.Compression.Gzip
    
    # Determine if connection should be insecure based on endpoint
    if endpoint.startswith("http://") or ":4317" in endpoint:
        kwargs["insecure"] = True
    else:
        kwargs["insecure"] = False
    
    if not api_key:
        raise ExporterConfigurationError("SPYGLASS_API_KEY is not set")
    
    # Set Authorization header with Bearer token
    # gRPC metadata keys must be lowercase
    kwargs["headers"] = {"authorization": f"Bearer {api_key}"}
    
    return OTLPSpanExporter(**kwargs)

# Create the tracer provider with resource attributes
resource = _create_resource()
provider = TracerProvider(resource=resource)
exporter = _create_exporter()
processor = BatchSpanProcessor(exporter)
provider.add_span_processor(processor)

# Sets the global default tracer provider
trace.set_tracer_provider(provider)

# Creates a tracer from the global tracer provider
spyglass_tracer = trace.get_tracer("spyglass-tracer")