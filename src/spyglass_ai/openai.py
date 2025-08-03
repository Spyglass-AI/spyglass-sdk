import functools

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer

# TODO: Implement wrappers the different client types (sync, async, streaming)
# TODO: Add metrics to track
#   - Number of calls to each type of endpoint
#   - Number of errors


def spyglass_openai(client_instance):
    """
    Wraps an OpenAI client instance to add tracing to chat completions.

    Args:
        client_instance: An OpenAI client instance (sync or async)

    Returns:
        The same client instance with tracing enabled
    """
    # Get a reference to the original method we want to wrap.
    original_create_method = client_instance.chat.completions.create

    @functools.wraps(original_create_method)
    def new_method_for_client(*args, **kwargs):
        # Start a new span
        with spyglass_tracer.start_as_current_span("openai.chat.completions.create") as span:
            try:
                # Set attributes for the OpenAI call
                # TODO: Double check these attributes
                if "model" in kwargs:
                    span.set_attribute("openai.model", kwargs["model"])
                if "messages" in kwargs:
                    span.set_attribute("openai.messages.count", len(kwargs["messages"]))
                if "max_tokens" in kwargs:
                    span.set_attribute("openai.max_tokens", kwargs["max_tokens"])
                if "temperature" in kwargs:
                    span.set_attribute("openai.temperature", kwargs["temperature"])

                # Call the original method
                result = original_create_method(*args, **kwargs)

                # Add response attributes if available
                if hasattr(result, "usage") and result.usage:
                    if hasattr(result.usage, "prompt_tokens"):
                        span.set_attribute(
                            "openai.usage.prompt_tokens",
                            result.usage.prompt_tokens,
                        )
                    if hasattr(result.usage, "completion_tokens"):
                        span.set_attribute(
                            "openai.usage.completion_tokens",
                            result.usage.completion_tokens,
                        )
                    if hasattr(result.usage, "total_tokens"):
                        span.set_attribute(
                            "openai.usage.total_tokens",
                            result.usage.total_tokens,
                        )

                # Get the model response and save it as an attribute
                if hasattr(result, "model"):
                    span.set_attribute("openai.response.model", result.model)

                # Set span status to OK for successful calls
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                # Record the exception
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey patch the method on the client instance with our wrapper method.
    client_instance.chat.completions.create = new_method_for_client

    return client_instance
