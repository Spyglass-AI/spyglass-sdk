import functools
import json
from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer

# TODO: Implement wrappers the different client types (sync, async, streaming)
# TODO: Add metrics to track
#   - Number of calls to each type of endpoint
#   - Number of errors
#
# This wrapper follows OpenTelemetry GenAI semantic conventions:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/


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
        # Set record_exception=False since we manually record exceptions in the except block
        with spyglass_tracer.start_as_current_span(
            "openai.chat.completions.create", record_exception=False
        ) as span:
            try:
                # Set OpenTelemetry GenAI semantic convention attributes
                span.set_attribute("gen_ai.operation.name", "chat")
                span.set_attribute("gen_ai.system", "openai")

                # Model information
                if "model" in kwargs:
                    span.set_attribute("gen_ai.request.model", kwargs["model"])

                # Request parameters
                if "max_tokens" in kwargs:
                    span.set_attribute(
                        "gen_ai.request.max_tokens", kwargs["max_tokens"]
                    )
                if "temperature" in kwargs:
                    span.set_attribute(
                        "gen_ai.request.temperature", kwargs["temperature"]
                    )
                if "top_p" in kwargs:
                    span.set_attribute("gen_ai.request.top_p", kwargs["top_p"])
                if "frequency_penalty" in kwargs:
                    span.set_attribute(
                        "gen_ai.request.frequency_penalty", kwargs["frequency_penalty"]
                    )
                if "presence_penalty" in kwargs:
                    span.set_attribute(
                        "gen_ai.request.presence_penalty", kwargs["presence_penalty"]
                    )

                # Messages and input content
                if "messages" in kwargs:
                    messages = kwargs["messages"]
                    span.set_attribute("gen_ai.input.messages.count", len(messages))

                    # Convert messages to the standard format for GenAI semantic conventions
                    formatted_messages = _format_openai_messages(messages)
                    span.set_attribute(
                        "gen_ai.input.messages", json.dumps(formatted_messages)
                    )

                # Tools information
                if "tools" in kwargs and kwargs["tools"]:
                    span.set_attribute(
                        "gen_ai.request.tools.count", len(kwargs["tools"])
                    )
                    tool_names = [
                        tool.get("function", {}).get("name", "unknown")
                        for tool in kwargs["tools"]
                    ]
                    span.set_attribute(
                        "gen_ai.request.tools.names", ",".join(tool_names)
                    )

                # Call the original method
                result = original_create_method(*args, **kwargs)

                # Set response attributes following GenAI semantic conventions
                if hasattr(result, "model"):
                    span.set_attribute("gen_ai.response.model", result.model)

                # Usage metadata
                if hasattr(result, "usage") and result.usage:
                    if hasattr(result.usage, "prompt_tokens"):
                        span.set_attribute(
                            "gen_ai.usage.input_tokens", result.usage.prompt_tokens
                        )
                    if hasattr(result.usage, "completion_tokens"):
                        span.set_attribute(
                            "gen_ai.usage.output_tokens", result.usage.completion_tokens
                        )
                    if hasattr(result.usage, "total_tokens"):
                        span.set_attribute(
                            "gen_ai.usage.total_tokens", result.usage.total_tokens
                        )

                # Response content and messages
                if hasattr(result, "choices") and result.choices:
                    span.set_attribute(
                        "gen_ai.response.choices.count", len(result.choices)
                    )

                    # Format and record response messages
                    response_messages = _format_openai_response(result.choices)
                    span.set_attribute(
                        "gen_ai.output.messages", json.dumps(response_messages)
                    )

                    # Record finish reasons
                    finish_reasons = [
                        choice.finish_reason
                        for choice in result.choices
                        if choice.finish_reason
                    ]
                    if finish_reasons:
                        span.set_attribute(
                            "gen_ai.response.finish_reasons", ",".join(finish_reasons)
                        )

                # Response metadata
                if hasattr(result, "id"):
                    span.set_attribute("gen_ai.response.id", result.id)
                if hasattr(result, "created"):
                    span.set_attribute("gen_ai.response.created", result.created)

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


def _format_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format OpenAI messages to GenAI semantic convention format."""
    formatted_messages = []

    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")

        formatted_message = {"role": role, "content": content}

        # Handle tool calls
        if "tool_calls" in message and message["tool_calls"]:
            formatted_message["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
                for tc in message["tool_calls"]
            ]

        # Handle tool call results
        if "tool_call_id" in message:
            formatted_message["tool_call_id"] = message["tool_call_id"]

        formatted_messages.append(formatted_message)

    return formatted_messages


def _format_openai_response(choices: List[Any]) -> List[Dict[str, Any]]:
    """Format OpenAI response choices to GenAI semantic convention format."""
    formatted_responses = []

    for choice in choices:
        if hasattr(choice, "message"):
            message = choice.message
            formatted_response = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", "") or "",
            }

            # Handle tool calls in response
            if hasattr(message, "tool_calls") and message.tool_calls:
                formatted_response["tool_calls"] = [
                    {
                        "id": tc.id if hasattr(tc, "id") else "",
                        "type": tc.type if hasattr(tc, "type") else "function",
                        "function": {
                            "name": tc.function.name
                            if hasattr(tc, "function") and hasattr(tc.function, "name")
                            else "",
                            "arguments": tc.function.arguments
                            if hasattr(tc, "function")
                            and hasattr(tc.function, "arguments")
                            else "",
                        },
                    }
                    for tc in message.tool_calls
                ]

            formatted_responses.append(formatted_response)

    return formatted_responses
