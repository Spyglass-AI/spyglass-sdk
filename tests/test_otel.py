import os
from unittest.mock import Mock, patch

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_tracer_provider_setup():
    """Test that the tracer provider is properly configured."""
    # Import here to ensure fresh setup
    from spyglass_ai.otel import provider, spyglass_tracer

    # Check that provider is a TracerProvider instance
    assert isinstance(provider, TracerProvider)

    # Check that tracer is created
    assert spyglass_tracer is not None
    assert spyglass_tracer.instrumentation_info.name == "spyglass-tracer"


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_span_processor_configured():
    """Test that the span processor is properly added."""
    from spyglass_ai.otel import provider

    # Check that provider has span processors
    assert len(provider._active_span_processor._span_processors) > 0

    # Check that one of the processors is BatchSpanProcessor
    processors = provider._active_span_processor._span_processors
    batch_processor_found = any(isinstance(proc, BatchSpanProcessor) for proc in processors)
    assert batch_processor_found


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_otlp_exporter_configured():
    """Test that OTLP exporter is configured in the processor."""
    from spyglass_ai.otel import processor

    # Check that the processor has an OTLP exporter
    assert isinstance(processor.span_exporter, OTLPSpanExporter)


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_global_tracer_provider_set():
    """Test that the global tracer provider is set."""
    current_provider = trace.get_tracer_provider()

    # The global provider should be set (not the default NoOpTracerProvider)
    assert not current_provider.__class__.__name__ == "NoOpTracerProvider"


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_tracer_creates_spans():
    """Test that the tracer can create spans."""
    from spyglass_ai.otel import spyglass_tracer

    with spyglass_tracer.start_as_current_span("test_span") as span:
        assert span is not None
        assert span.name == "test_span"
        span.set_attribute("test_key", "test_value")

    # Span should be ended after context exit
    assert not span.is_recording()


# Tests for _create_exporter method


@patch.dict(os.environ, {}, clear=True)
def test_create_exporter_missing_api_key():
    """Test _create_exporter raises ExporterConfigurationError for missing API key."""
    from spyglass_ai.otel import ExporterConfigurationError, _create_exporter

    with pytest.raises(
        ExporterConfigurationError, match="SPYGLASS_API_KEY is required but not set"
    ):
        _create_exporter()


@patch.dict(os.environ, {"SPYGLASS_API_KEY": "test-api-key"})
@patch("spyglass_ai.otel.OTLPSpanExporter")
def test_create_exporter_default_config(mock_otlp_exporter):
    """Test that _create_exporter returns OTLPSpanExporter with default config."""
    from spyglass_ai.otel import _create_exporter

    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance

    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        endpoint="https://ingest.spyglass-ai.com/v1/traces",
        headers={"Authorization": "Bearer test-api-key"},
    )


@patch.dict(
    os.environ,
    {
        "SPYGLASS_API_KEY": "test-api-key",
        "SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318/v1/traces",
    },
)
@patch("spyglass_ai.otel.OTLPSpanExporter")
def test_create_exporter_with_custom_endpoint(mock_otlp_exporter):
    """Test _create_exporter configures custom endpoint from env variable."""
    from spyglass_ai.otel import _create_exporter

    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance

    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        endpoint="http://localhost:4318/v1/traces",
        headers={"Authorization": "Bearer test-api-key"},
    )


# Tests for _create_resource method


@patch.dict(os.environ, {}, clear=True)
def test_create_resource_missing_deployment_id():
    """Test _create_resource raises DeploymentConfigurationError for missing deployment ID."""
    from spyglass_ai.otel import DeploymentConfigurationError, _create_resource

    with pytest.raises(
        DeploymentConfigurationError,
        match="SPYGLASS_DEPLOYMENT_ID is required but not set",
    ):
        _create_resource()


@patch.dict(os.environ, {"SPYGLASS_DEPLOYMENT_ID": "test-deployment-123"})
def test_create_resource_success():
    """Test that _create_resource creates resource with correct attributes."""
    from spyglass_ai.otel import _create_resource

    resource = _create_resource()

    # Check that resource has correct attributes
    assert resource.attributes["service.name"] == "test-deployment-123"
    assert resource.attributes["deployment.id"] == "test-deployment-123"


@patch.dict(
    os.environ,
    {"SPYGLASS_DEPLOYMENT_ID": "test-deployment", "SPYGLASS_API_KEY": "test-api-key"},
)
def test_exporter_integration():
    """Test that the module-level exporter is properly created and used."""
    from spyglass_ai.otel import exporter, processor

    # The exporter should be created by _create_exporter()
    assert exporter is not None
    assert isinstance(exporter, OTLPSpanExporter)

    # The processor should use the created exporter
    assert processor.span_exporter == exporter


def test_exception_hierarchy():
    """Test that custom exceptions have proper inheritance."""
    from spyglass_ai.otel import (
        DeploymentConfigurationError,
        ExporterConfigurationError,
        SpyglassOtelError,
    )

    # Test exception hierarchy
    assert issubclass(ExporterConfigurationError, SpyglassOtelError)
    assert issubclass(DeploymentConfigurationError, SpyglassOtelError)
    assert issubclass(SpyglassOtelError, Exception)

    # Test that exceptions can be instantiated
    base_error = SpyglassOtelError("base error")
    exporter_error = ExporterConfigurationError("exporter error")
    deployment_error = DeploymentConfigurationError("deployment error")

    assert str(base_error) == "base error"
    assert str(exporter_error) == "exporter error"
    assert str(deployment_error) == "deployment error"
