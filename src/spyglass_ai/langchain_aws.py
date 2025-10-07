import functools
import json
from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer


def spyglass_chatbedrockconverse(llm_instance):
    """
    Wraps a ChatBedrockConverse instance to add comprehensive tracing.

    This wrapper follows OpenTelemetry GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    Args:
        llm_instance: A ChatBedrockConverse instance

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

    @functools.wraps(original_generate)
    def traced_generate(messages, stop=None, run_manager=None, **kwargs):
        # Set record_exception=False since we manually record exceptions in the except block
        with spyglass_tracer.start_as_current_span(
            "bedrock.chat.generate", record_exception=False
        ) as span:
            try:
                # Set Bedrock-specific attributes
                _set_bedrock_attributes(span, llm_instance, messages, kwargs)

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

    llm_instance._generate = traced_generate


def _set_bedrock_attributes(span, llm_instance, messages, kwargs):
    """Set span attributes following GenAI semantic conventions"""
    # GenAI semantic convention attributes
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", "aws_bedrock")
    span.set_attribute("gen_ai.request.model", llm_instance.model_id)

    # AWS Bedrock specific attributes (using gen_ai prefix for consistency)
    span.set_attribute(
        "gen_ai.request.aws.region", llm_instance.region_name or "us-east-1"
    )
    span.set_attribute("gen_ai.request.aws.provider", llm_instance.provider)

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
                if "toolSpec" in tool:
                    tool_names.append(tool["toolSpec"].get("name", "unknown"))
                elif "function" in tool:
                    tool_names.append(tool["function"].get("name", "unknown"))
        if tool_names:
            span.set_attribute("gen_ai.request.tools.names", ",".join(tool_names))

    # AWS Bedrock specific configurations
    if llm_instance.guardrail_config:
        span.set_attribute("gen_ai.request.aws.guardrails.enabled", True)

    if llm_instance.performance_config:
        span.set_attribute("gen_ai.request.aws.performance_config.enabled", True)


def _set_response_attributes(span, result):
    """Set response-specific attributes following GenAI semantic conventions"""
    if hasattr(result, "generations") and result.generations:
        generation = result.generations[0]
        message = generation.message

        # Usage metadata (LangChain extracts from Converse "usage")
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage = message.usage_metadata

            # Handle the actual format returned by LangChain AWS
            if isinstance(usage, dict):
                # Dictionary format - use keys directly
                if "input_tokens" in usage:
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", usage["input_tokens"]
                    )
                if "output_tokens" in usage:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", usage["output_tokens"]
                    )
                if "total_tokens" in usage:
                    span.set_attribute(
                        "gen_ai.usage.total_tokens", usage["total_tokens"]
                    )

                # AWS Bedrock cache tokens (dict format)
                if "input_token_details" in usage and isinstance(
                    usage["input_token_details"], dict
                ):
                    details = usage["input_token_details"]
                    if "cache_read" in details:
                        span.set_attribute(
                            "gen_ai.usage.aws.cache_read_tokens", details["cache_read"]
                        )
                    if "cache_creation" in details:
                        span.set_attribute(
                            "gen_ai.usage.aws.cache_write_tokens",
                            details["cache_creation"],
                        )
            else:
                # Object format - use attributes
                span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
                span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)

                # AWS Bedrock cache tokens (object format)
                if hasattr(usage, "input_token_details") and usage.input_token_details:
                    details = usage.input_token_details
                    if "cache_read" in details:
                        span.set_attribute(
                            "gen_ai.usage.aws.cache_read_tokens", details["cache_read"]
                        )
                    if "cache_creation" in details:
                        span.set_attribute(
                            "gen_ai.usage.aws.cache_write_tokens",
                            details["cache_creation"],
                        )

        # Format and record output messages
        formatted_output = _format_langchain_messages([message])
        span.set_attribute("gen_ai.output.messages", json.dumps(formatted_output))

        # Tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            span.set_attribute("gen_ai.response.tools.count", len(message.tool_calls))
            tool_names = [tc.get("name", "unknown") for tc in message.tool_calls]
            span.set_attribute("gen_ai.response.tools.names", ",".join(tool_names))

        # Response metadata (LangChain adds model_name and preserves Converse response fields)
        if hasattr(message, "response_metadata") and message.response_metadata:
            metadata = message.response_metadata

            # Model name (LangChain always sets this)
            if "model_name" in metadata:
                span.set_attribute("gen_ai.response.model", metadata["model_name"])

            # Stop reason from Converse API
            if "stopReason" in metadata:
                span.set_attribute(
                    "gen_ai.response.finish_reasons", metadata["stopReason"]
                )

            # Latency metrics (LangChain converts to list format)
            if "metrics" in metadata and "latencyMs" in metadata["metrics"]:
                latency = metadata["metrics"]["latencyMs"]
                # LangChain converts single values to lists, so take first element
                if isinstance(latency, list) and latency:
                    span.set_attribute("gen_ai.response.aws.latency_ms", latency[0])
                elif isinstance(latency, (int, float)):
                    span.set_attribute("gen_ai.response.aws.latency_ms", latency)


def _wrap_async_methods(llm_instance):
    """Wrap async methods if they exist (inherited from base class)"""
    # Check if async methods are available and wrap them
    if hasattr(llm_instance, "_agenerate"):
        _wrap_agenerate_method(llm_instance)


def _wrap_agenerate_method(llm_instance):
    """Wrap the _agenerate method for async invocations"""
    original_agenerate = llm_instance._agenerate

    @functools.wraps(original_agenerate)
    async def traced_agenerate(messages, stop=None, run_manager=None, **kwargs):
        # Set record_exception=False since we manually record exceptions in the except block
        with spyglass_tracer.start_as_current_span(
            "bedrock.chat.agenerate", record_exception=False
        ) as span:
            try:
                # Set Bedrock-specific attributes
                _set_bedrock_attributes(span, llm_instance, messages, kwargs)

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
