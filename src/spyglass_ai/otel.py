import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


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
        raise DeploymentConfigurationError(
            "SPYGLASS_DEPLOYMENT_ID is required but not set"
        )

    # Use deployment_id for both service.name and deployment.id
    resource_attributes["service.name"] = deployment_id
    resource_attributes["deployment.id"] = deployment_id

    return Resource.create(resource_attributes)


def _create_exporter():
    """Create and return an OTLP HTTP span exporter.

    Uses SPYGLASS_API_KEY and SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT env vars.
    """
    api_key = os.getenv("SPYGLASS_API_KEY")

    # Check for custom endpoint (for development)
    endpoint = os.getenv(
        "SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://ingest.spyglass-ai.com/v1/traces",
    )

    kwargs = {}
    kwargs["endpoint"] = endpoint

    if not api_key:
        raise ExporterConfigurationError("SPYGLASS_API_KEY is required but not set")

    # Set Authorization header with Bearer token
    kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    exporter = OTLPSpanExporter(**kwargs)
    return exporter


# Global variables for lazy initialization
_spyglass_tracer = None


def get_spyglass_tracer():
    """Get the Spyglass tracer, initializing it if necessary."""
    global _spyglass_tracer

    if _spyglass_tracer is not None:
        return _spyglass_tracer

    # Create the tracer provider with resource attributes
    resource = _create_resource()
    provider = TracerProvider(resource=resource)
    exporter = _create_exporter()
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Sets the global default tracer provider
    trace.set_tracer_provider(provider)

    # Create and cache the tracer
    _spyglass_tracer = trace.get_tracer("spyglass-tracer")

    return _spyglass_tracer


# For backward compatibility, create the tracer attribute that gets initialized lazily
class _LazyTracer:
    def __getattr__(self, name):
        tracer = get_spyglass_tracer()
        return getattr(tracer, name)


spyglass_tracer = _LazyTracer()
