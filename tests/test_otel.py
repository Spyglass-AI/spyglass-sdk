import os
import pytest
from unittest.mock import Mock, patch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def test_tracer_provider_setup():
    """Test that the tracer provider is properly configured."""
    # Import here to ensure fresh setup
    from spyglass_ai.otel import provider, spyglass_tracer
    
    # Check that provider is a TracerProvider instance
    assert isinstance(provider, TracerProvider)
    
    # Check that tracer is created
    assert spyglass_tracer is not None
    assert spyglass_tracer.instrumentation_info.name == "spyglass-tracer"

def test_span_processor_configured():
    """Test that the span processor is properly added."""
    from spyglass_ai.otel import provider
    
    # Check that provider has span processors
    assert len(provider._active_span_processor._span_processors) > 0
    
    # Check that one of the processors is BatchSpanProcessor
    processors = provider._active_span_processor._span_processors
    batch_processor_found = any(
        isinstance(proc, BatchSpanProcessor) for proc in processors
    )
    assert batch_processor_found

def test_otlp_exporter_configured():
    """Test that OTLP exporter is configured in the processor."""
    from spyglass_ai.otel import processor, OTLPSpanExporter
    
    # Check that the processor has an OTLP exporter
    assert isinstance(processor.span_exporter, OTLPSpanExporter)

def test_global_tracer_provider_set():
    """Test that the global tracer provider is set."""
    current_provider = trace.get_tracer_provider()
    
    # The global provider should be set (not the default NoOpTracerProvider)
    assert not current_provider.__class__.__name__ == "NoOpTracerProvider"

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
@patch('spyglass_ai.otel.OTLPSpanExporter')
def test_create_exporter_default_config(mock_otlp_exporter):
    """Test that _create_exporter returns OTLPSpanExporter with default config."""
    from spyglass_ai.otel import _create_exporter
    
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with()

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"
})
@patch('spyglass_ai.otel.OTLPSpanExporter')
def test_create_exporter_with_endpoint(mock_otlp_exporter):
    """Test that _create_exporter configures endpoint from environment variable."""
    from spyglass_ai.otel import _create_exporter
    
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(endpoint="http://localhost:4317")

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_HEADERS": "authorization=Bearer token,x-custom=value"
})
@patch('spyglass_ai.otel.OTLPSpanExporter')
def test_create_exporter_with_headers(mock_otlp_exporter):
    """Test that _create_exporter configures headers from environment variable."""
    from spyglass_ai.otel import _create_exporter
    
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        headers={"authorization": "Bearer token", "x-custom": "value"}
    )

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
    "SPYGLASS_OTEL_EXPORTER_OTLP_HEADERS": "authorization=Bearer token,x-custom=value"
})
@patch('spyglass_ai.otel.OTLPSpanExporter')
def test_create_exporter_with_full_config(mock_otlp_exporter):
    """Test that _create_exporter configures both endpoint and headers."""
    from spyglass_ai.otel import _create_exporter
    
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        endpoint="http://localhost:4317",
        headers={"authorization": "Bearer token", "x-custom": "value"}
    )

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_HEADERS": "malformed-header"
})
def test_create_exporter_malformed_headers():
    """Test that _create_exporter raises ExporterConfigurationError for malformed headers."""
    from spyglass_ai.otel import _create_exporter, ExporterConfigurationError
    
    with pytest.raises(ExporterConfigurationError) as exc_info:
        _create_exporter()
    
    assert "Invalid headers format" in str(exc_info.value)
    assert "malformed-header" in str(exc_info.value)

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_HEADERS": "key1=value1,malformed,key2=value2"
})
def test_create_exporter_partial_malformed_headers():
    """Test that _create_exporter raises ExporterConfigurationError for any malformed headers."""
    from spyglass_ai.otel import _create_exporter, ExporterConfigurationError
    
    with pytest.raises(ExporterConfigurationError) as exc_info:
        _create_exporter()
    
    assert "Invalid headers format" in str(exc_info.value)
    assert "key1=value1,malformed,key2=value2" in str(exc_info.value)

@patch.dict(os.environ, {
    "SPYGLASS_OTEL_EXPORTER_OTLP_HEADERS": " key1 = value1 , key2 = value2 "
})
@patch('spyglass_ai.otel.OTLPSpanExporter')
def test_create_exporter_headers_with_spaces(mock_otlp_exporter):
    """Test that _create_exporter properly trims spaces from headers."""
    from spyglass_ai.otel import _create_exporter
    
    mock_exporter_instance = Mock()
    mock_otlp_exporter.return_value = mock_exporter_instance
    
    exporter = _create_exporter()
    assert exporter == mock_exporter_instance
    mock_otlp_exporter.assert_called_once_with(
        headers={"key1": "value1", "key2": "value2"}
    )

def test_exporter_integration():
    """Test that the module-level exporter is properly created and used."""
    from spyglass_ai.otel import exporter, processor, OTLPSpanExporter
    
    # The exporter should be created by _create_exporter()
    assert exporter is not None
    assert isinstance(exporter, OTLPSpanExporter)
    
    # The processor should use the created exporter
    assert processor.span_exporter == exporter

def test_exception_hierarchy():
    """Test that custom exceptions have proper inheritance."""
    from spyglass_ai.otel import SpyglassOtelError, ExporterConfigurationError
    
    # Test exception hierarchy
    assert issubclass(ExporterConfigurationError, SpyglassOtelError)
    assert issubclass(SpyglassOtelError, Exception)
    
    # Test that exceptions can be instantiated
    base_error = SpyglassOtelError("base error")
    config_error = ExporterConfigurationError("config error")
    
    assert str(base_error) == "base error"
    assert str(config_error) == "config error"
