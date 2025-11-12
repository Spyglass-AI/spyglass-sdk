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


# Tests for programmatic configuration


@patch.dict(os.environ, {}, clear=True)
@patch("spyglass_ai.otel.OTLPSpanExporter")
def test_configure_programmatic_config(mock_otlp_exporter):
    """Test that configure_spyglass() allows programmatic configuration."""
    from spyglass_ai.otel import (
        configure_spyglass,
        _create_exporter,
        _create_resource,
        _config,
    )

    # Reset any existing config by clearing the module-level dict
    _config["api_key"] = None
    _config["deployment_id"] = None
    _config["endpoint"] = None

    # Configure programmatically
    configure_spyglass(
        api_key="programmatic-api-key",
        deployment_id="programmatic-deployment",
        endpoint="https://custom-endpoint.com/v1/traces",
    )

    # Test exporter uses programmatic config
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance

    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        endpoint="https://custom-endpoint.com/v1/traces",
        headers={"Authorization": "Bearer programmatic-api-key"},
    )

    # Test resource uses programmatic config
    resource = _create_resource()
    assert resource.attributes["service.name"] == "programmatic-deployment"
    assert resource.attributes["deployment.id"] == "programmatic-deployment"


@patch.dict(
    os.environ,
    {"SPYGLASS_API_KEY": "env-api-key", "SPYGLASS_DEPLOYMENT_ID": "env-deployment"},
)
@patch("spyglass_ai.otel.OTLPSpanExporter")
def test_configure_takes_precedence_over_env_vars(mock_otlp_exporter):
    """Test that programmatic config takes precedence over environment variables."""
    from spyglass_ai.otel import (
        configure_spyglass,
        _create_exporter,
        _create_resource,
        _config,
    )

    # Reset any existing config by clearing the module-level dict
    _config["api_key"] = None
    _config["deployment_id"] = None
    _config["endpoint"] = None

    # Configure programmatically - should override env vars
    configure_spyglass(
        api_key="programmatic-api-key",
        deployment_id="programmatic-deployment",
    )

    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance

    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    # Should use programmatic config, not env vars
    mock_otlp_exporter.assert_called_once_with(
        endpoint="https://ingest.spyglass-ai.com/v1/traces",  # Default endpoint
        headers={"Authorization": "Bearer programmatic-api-key"},
    )

    resource = _create_resource()
    assert resource.attributes["service.name"] == "programmatic-deployment"
    assert resource.attributes["deployment.id"] == "programmatic-deployment"


@patch.dict(
    os.environ,
    {"SPYGLASS_API_KEY": "env-api-key", "SPYGLASS_DEPLOYMENT_ID": "env-deployment"},
)
@patch("spyglass_ai.otel.OTLPSpanExporter")
def test_env_vars_fallback_when_not_configured(mock_otlp_exporter):
    """Test that environment variables are used when programmatic config is not set."""
    from spyglass_ai.otel import (
        configure_spyglass,
        _create_exporter,
        _create_resource,
        _config,
    )

    # Reset any existing config by clearing the module-level dict
    _config["api_key"] = None
    _config["deployment_id"] = None
    _config["endpoint"] = None

    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance

    # Should fall back to env vars
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        endpoint="https://ingest.spyglass-ai.com/v1/traces",
        headers={"Authorization": "Bearer env-api-key"},
    )

    resource = _create_resource()
    assert resource.attributes["service.name"] == "env-deployment"
    assert resource.attributes["deployment.id"] == "env-deployment"


@patch.dict(os.environ, {}, clear=True)
def test_configure_resets_tracer():
    """Test that configure_spyglass() resets the tracer so it reinitializes with new config."""
    from spyglass_ai.otel import configure_spyglass, get_spyglass_tracer, _config

    # Reset any existing config by clearing the module-level dict
    _config["api_key"] = None
    _config["deployment_id"] = None
    _config["endpoint"] = None

    # Configure and get tracer
    configure_spyglass(api_key="key1", deployment_id="deployment1")
    tracer1 = get_spyglass_tracer()

    # Reconfigure and get tracer again - should be reinitialized
    configure_spyglass(api_key="key2", deployment_id="deployment2")
    tracer2 = get_spyglass_tracer()

    # Tracers should be different instances (reinitialized)
    assert tracer1 is not tracer2
