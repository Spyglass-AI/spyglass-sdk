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


# Module-level configuration storage for programmatic configuration
_config = {
    "api_key": None,
    "deployment_id": None,
    "endpoint": None,
}


def configure_spyglass(
    api_key: str = None,
    deployment_id: str = None,
    endpoint: str = None,
):
    """Configure Spyglass SDK programmatically.

    This function allows you to configure the Spyglass SDK without using
    environment variables. Configuration values passed here take precedence
    over environment variables.

    Args:
        api_key: Spyglass API key (required if not set via SPYGLASS_API_KEY env var).
                 Pass None to clear programmatic config and use env var instead.
        deployment_id: Deployment identifier (required if not set via SPYGLASS_DEPLOYMENT_ID env var).
                       Pass None to clear programmatic config and use env var instead.
        endpoint: Optional custom OTLP endpoint (overrides SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT).
                  Pass None to clear programmatic config and use env var instead.

    Example:
        ```python
        from spyglass_ai import configure_spyglass

        configure_spyglass(
            api_key="your-api-key",
            deployment_id="my-service-v1.0.0",
            endpoint="https://custom-endpoint.com/v1/traces"  # Optional
        )
        ```

    Note:
        If the tracer has already been initialized, calling configure_spyglass() will reset it
        so it reinitializes with the new configuration. The tracer is initialized
        lazily on first use after configuration.
    """
    global _config, _spyglass_tracer

    # Update config for provided non-None values
    if api_key is not None:
        _config["api_key"] = api_key
    if deployment_id is not None:
        _config["deployment_id"] = deployment_id
    if endpoint is not None:
        _config["endpoint"] = endpoint

    # Reset tracer so it reinitializes with new config
    _spyglass_tracer = None


def _create_resource():
    """Create and return a Resource with deployment and service information."""
    resource_attributes = {}

    # Check programmatic config first, then fall back to env vars
    deployment_id = _config["deployment_id"] or os.getenv("SPYGLASS_DEPLOYMENT_ID")
    if not deployment_id:
        raise DeploymentConfigurationError(
            "SPYGLASS_DEPLOYMENT_ID is required but not set. "
            "Set it via configure_spyglass() or SPYGLASS_DEPLOYMENT_ID environment variable."
        )

    # Use deployment_id for both service.name and deployment.id
    resource_attributes["service.name"] = deployment_id
    resource_attributes["deployment.id"] = deployment_id

    return Resource.create(resource_attributes)


def _create_exporter():
    """Create and return an OTLP HTTP span exporter.

    Uses programmatic config if set, otherwise falls back to
    SPYGLASS_API_KEY and SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT env vars.
    """
    # Check programmatic config first, then fall back to env vars
    api_key = _config["api_key"] or os.getenv("SPYGLASS_API_KEY")

    # Check for custom endpoint (programmatic config takes precedence)
    endpoint = _config["endpoint"] or os.getenv(
        "SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT",
        "https://ingest.spyglass-ai.com/v1/traces",
    )

    if not api_key:
        raise ExporterConfigurationError(
            "SPYGLASS_API_KEY is required but not set. "
            "Set it via configure_spyglass() or SPYGLASS_API_KEY environment variable."
        )

    kwargs = {
        "endpoint": endpoint,
        "headers": {"Authorization": f"Bearer {api_key}"},
    }

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
