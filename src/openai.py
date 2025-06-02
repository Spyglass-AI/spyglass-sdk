import functools
from otel import spyglass_tracer

# TODO: Implement wrappers the different client types (sync, async, streaming)

def spyglass_openai(client_instance):
    # Get a reference to the original method we want to wrap.
    original_create_method = client_instance.chat.completions.create
    
    @functools.wraps(original_create_method)
    def new_method_for_client(*args, **kwargs):        
        # Start a new span
        with spyglass_tracer.start_as_current_span("chat.completions.create_with_otel") as span:
            spyglass_tracer.set_span_attributes(span, {"model": kwargs.get("model")})            
            decorated_call = spyglass_tracer(original_create_method, "chat.completions.create")
            try:
                result = decorated_call(*args, **kwargs)
                return result
            except Exception as e:
                raise

    # Monkey patch the method on the client instance with our wrapper method.
    client_instance.chat.completions.create = new_method_for_client
    
    return client_instance