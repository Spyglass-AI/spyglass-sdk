import json
from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer


def spyglass_chatopenai(llm_instance):
    """
    Wraps a ChatOpenAI instance to add comprehensive tracing.

    This wrapper follows OpenTelemetry GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    Args:
        llm_instance: A ChatOpenAI instance

    Returns:
        The same instance with tracing enabled
    """
    # Wrap the core generation methods
    _wrap_generate_method(llm_instance)

    # Wrap async methods if available
    _wrap_async_methods(llm_instance)

    return llm_instance


def _wrap_generate_method(llm_instance):
    """Wrap the _generate method for sync invocations"""
    original_generate = llm_instance._generate

    def traced_generate(messages, stop=None, run_manager=None, **kwargs):
        # Set record_exception=False since we manually record exceptions in the except block
        with spyglass_tracer.start_as_current_span(
            "openai.chat.generate", record_exception=False
        ) as span:
            try:
                # Set OpenAI-specific attributes
                _set_openai_attributes(span, llm_instance, messages, kwargs)

                # Call original method
                result = original_generate(messages, stop, run_manager, **kwargs)

                # Extract response attributes
                _set_response_attributes(span, result)

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Safely wrap function metadata
    try:
        traced_generate.__name__ = getattr(original_generate, "__name__", "traced_generate")
        traced_generate.__doc__ = getattr(original_generate, "__doc__", None)
        traced_generate.__module__ = getattr(original_generate, "__module__", None)
        traced_generate.__qualname__ = getattr(original_generate, "__qualname__", "traced_generate")
    except (TypeError, AttributeError):
        pass  # If copying fails, continue without metadata

    llm_instance._generate = traced_generate


def _set_openai_attributes(span, llm_instance, messages, kwargs):
    """Set span attributes following GenAI semantic conventions"""
    # GenAI semantic convention attributes
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", llm_instance.model_name)

    # Model parameters (GenAI semantic conventions)
    if llm_instance.temperature is not None:
        span.set_attribute("gen_ai.request.temperature", llm_instance.temperature)
    if llm_instance.max_tokens is not None:
        span.set_attribute("gen_ai.request.max_tokens", llm_instance.max_tokens)
    if llm_instance.top_p is not None:
        span.set_attribute("gen_ai.request.top_p", llm_instance.top_p)

    # Message information (GenAI semantic conventions)
    span.set_attribute("gen_ai.input.messages.count", len(messages))

    # Format and record input messages
    formatted_messages = _format_langchain_messages(messages)
    span.set_attribute("gen_ai.input.messages", json.dumps(formatted_messages))

    # Tool information from kwargs (when tools are bound)
    if "tools" in kwargs:
        span.set_attribute("gen_ai.request.tools.count", len(kwargs["tools"]))
        tool_names = []
        for tool in kwargs["tools"]:
            if isinstance(tool, dict):
                if "function" in tool:
                    tool_names.append(tool["function"].get("name", "unknown"))
                elif "name" in tool:
                    tool_names.append(tool.get("name", "unknown"))
        if tool_names:
            span.set_attribute("gen_ai.request.tools.names", ",".join(tool_names))


def _set_response_attributes(span, result):
    """Set response-specific attributes following GenAI semantic conventions"""
    if hasattr(result, "generations") and result.generations:
        generation = result.generations[0]
        message = generation.message

        # Usage metadata (UsageMetadata is a TypedDict from langchain_core.messages.ai)
        # Primary source: message.usage_metadata (always a dict when present)
        usage = None
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage = message.usage_metadata
        
        # Fallback: check llm_output.token_usage if usage_metadata is not available
        if not usage and hasattr(result, "llm_output") and result.llm_output:
            token_usage = result.llm_output.get("token_usage")
            if token_usage:
                # Convert OpenAI's token_usage format to UsageMetadata format
                usage = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
        
        # Set usage attributes (UsageMetadata is a TypedDict, accessed as dict)
        if usage and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")
            
            if input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            if output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            if total_tokens is not None:
                span.set_attribute("gen_ai.usage.total_tokens", total_tokens)

        # Format and record output messages
        formatted_output = _format_langchain_messages([message])
        span.set_attribute("gen_ai.output.messages", json.dumps(formatted_output))

        # Tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            span.set_attribute("gen_ai.response.tools.count", len(message.tool_calls))
            tool_names = [tc.get("name", "unknown") for tc in message.tool_calls]
            span.set_attribute("gen_ai.response.tools.names", ",".join(tool_names))

        # Response metadata
        if hasattr(message, "response_metadata") and message.response_metadata:
            metadata = message.response_metadata

            # Model name
            if "model_name" in metadata:
                span.set_attribute("gen_ai.response.model", metadata["model_name"])

            # Finish reason
            if "finish_reason" in metadata:
                span.set_attribute("gen_ai.response.finish_reasons", metadata["finish_reason"])


def _wrap_async_methods(llm_instance):
    """Wrap async methods if they exist"""
    if hasattr(llm_instance, "_agenerate"):
        _wrap_agenerate_method(llm_instance)


def _wrap_agenerate_method(llm_instance):
    """Wrap the _agenerate method for async invocations"""
    original_agenerate = llm_instance._agenerate

    async def traced_agenerate(messages, stop=None, run_manager=None, **kwargs):
        # Set record_exception=False since we manually record exceptions in the except block
        with spyglass_tracer.start_as_current_span(
            "openai.chat.agenerate", record_exception=False
        ) as span:
            try:
                # Set OpenAI-specific attributes
                _set_openai_attributes(span, llm_instance, messages, kwargs)

                # Call original method
                result = await original_agenerate(messages, stop, run_manager, **kwargs)

                # Extract response attributes
                _set_response_attributes(span, result)

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Safely wrap function metadata
    try:
        traced_agenerate.__name__ = getattr(original_agenerate, "__name__", "traced_agenerate")
        traced_agenerate.__doc__ = getattr(original_agenerate, "__doc__", None)
        traced_agenerate.__module__ = getattr(original_agenerate, "__module__", None)
        traced_agenerate.__qualname__ = getattr(original_agenerate, "__qualname__", "traced_agenerate")
    except (TypeError, AttributeError):
        pass  # If copying fails, continue without metadata

    llm_instance._agenerate = traced_agenerate


def _format_langchain_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Format LangChain messages (input or output) to GenAI semantic convention format."""
    formatted_messages = []

    for message in messages:
        # Extract role and content from LangChain message objects
        if hasattr(message, "__class__"):
            message_type = message.__class__.__name__.lower()
            if "human" in message_type or "user" in message_type:
                role = "user"
            elif "ai" in message_type or "assistant" in message_type:
                role = "assistant"
            elif "system" in message_type:
                role = "system"
            elif "tool" in message_type:
                role = "tool"
            else:
                role = "unknown"
        else:
            role = "unknown"

        # Extract content
        content = ""
        if hasattr(message, "content"):
            if isinstance(message.content, str):
                content = message.content
            elif isinstance(message.content, list):
                # Handle complex content like images, etc.
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)

        formatted_message = {"role": role, "content": content}

        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            formatted_message["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {}))
                        if tc.get("args")
                        else "",
                    },
                }
                for tc in message.tool_calls
            ]

        # Handle tool call results
        if hasattr(message, "tool_call_id"):
            formatted_message["tool_call_id"] = message.tool_call_id

        formatted_messages.append(formatted_message)

    return formatted_messages

