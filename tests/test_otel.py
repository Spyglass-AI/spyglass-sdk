from unittest.mock import Mock, patch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

def test_tracer_provider_setup():
    """Test that the tracer provider is properly configured."""
    # Import here to ensure fresh setup
    from src.otel import provider, spyglass_tracer
    
    # Check that provider is a TracerProvider instance
    assert isinstance(provider, TracerProvider)
    
    # Check that tracer is created
    assert spyglass_tracer is not None
    assert spyglass_tracer.instrumentation_info.name == "spyglass-tracer"

def test_span_processor_configured():
    """Test that the span processor is properly added."""
    from src.otel import provider
    
    # Check that provider has span processors
    assert len(provider._active_span_processor._span_processors) > 0
    
    # Check that one of the processors is BatchSpanProcessor
    processors = provider._active_span_processor._span_processors
    batch_processor_found = any(
        isinstance(proc, BatchSpanProcessor) for proc in processors
    )
    assert batch_processor_found

# TODO: Modify this test so that it checks that the span processor is 
# configured with the correct exporter
def test_console_exporter_configured():
    """Test that console exporter is configured in the processor."""
    from src.otel import processor
    
    # Check that the processor has a console exporter
    assert isinstance(processor.span_exporter, ConsoleSpanExporter)

def test_global_tracer_provider_set():
    """Test that the global tracer provider is set."""
    current_provider = trace.get_tracer_provider()
    
    # The global provider should be set (not the default NoOpTracerProvider)
    assert not current_provider.__class__.__name__ == "NoOpTracerProvider"

def test_tracer_creates_spans():
    """Test that the tracer can create spans."""
    from src.otel import spyglass_tracer
    
    with spyglass_tracer.start_as_current_span("test_span") as span:
        assert span is not None
        assert span.name == "test_span"
        span.set_attribute("test_key", "test_value")
        
    # Span should be ended after context exit
    assert not span.is_recording()
