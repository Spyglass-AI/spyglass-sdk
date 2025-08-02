import pytest
from opentelemetry import trace


@pytest.fixture(scope="session", autouse=True)
def cleanup_otel():
    """Ensure OpenTelemetry tracer provider is properly shut down after tests."""
    yield
    # Shutdown the tracer provider to stop background threads
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, "shutdown"):
        tracer_provider.shutdown()
