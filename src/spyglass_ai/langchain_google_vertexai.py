import functools
import json
from typing import Any, Dict, List

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer


def spyglass_chatvertexai(llm_instance):
    """
    Wraps a ChatVertexAI instance to add comprehensive tracing.

    This wrapper follows OpenTelemetry GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    Args:
        llm_instance: A ChatVertexAI instance from langchain-google-vertexai

    Returns:
        The same instance with tracing enabled
    """
    _wrap_generate_method(llm_instance)
    _wrap_async_methods(llm_instance)
    return llm_instance


def _wrap_generate_method(llm_instance):
    """Wrap the _generate method for sync invocations"""
    original_generate = llm_instance._generate

    @functools.wraps(original_generate)
    def traced_generate(messages, stop=None, run_manager=None, **kwargs):
        with spyglass_tracer.start_as_current_span(
            "vertexai.chat.generate", record_exception=False
        ) as span:
            try:
                _set_vertexai_attributes(span, llm_instance, messages, kwargs)

                result = original_generate(messages, stop, run_manager, **kwargs)

                _set_response_attributes(span, result)

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    llm_instance._generate = traced_generate


def _set_vertexai_attributes(span, llm_instance, messages, kwargs):
    """Set span attributes following GenAI semantic conventions"""
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.system", "vertex_ai")
    span.set_attribute("gen_ai.request.model", llm_instance.model_name)

    # Vertex AI specific attributes
    if hasattr(llm_instance, "project") and llm_instance.project:
        span.set_attribute("gen_ai.request.vertex_ai.project", llm_instance.project)
    if hasattr(llm_instance, "location") and llm_instance.location:
        span.set_attribute("gen_ai.request.vertex_ai.location", llm_instance.location)

    # Model parameters
    if hasattr(llm_instance, "temperature") and llm_instance.temperature is not None:
        span.set_attribute("gen_ai.request.temperature", llm_instance.temperature)
    if hasattr(llm_instance, "max_output_tokens") and llm_instance.max_output_tokens is not None:
        span.set_attribute("gen_ai.request.max_tokens", llm_instance.max_output_tokens)
    if hasattr(llm_instance, "top_p") and llm_instance.top_p is not None:
        span.set_attribute("gen_ai.request.top_p", llm_instance.top_p)
    if hasattr(llm_instance, "top_k") and llm_instance.top_k is not None:
        span.set_attribute("gen_ai.request.top_k", llm_instance.top_k)

    # Message information
    span.set_attribute("gen_ai.input.messages.count", len(messages))

    # Format and record input messages
    formatted_messages = _format_langchain_messages(messages)
    span.set_attribute("gen_ai.input.messages", json.dumps(formatted_messages))

    # Tool information from kwargs
    if "tools" in kwargs:
        span.set_attribute("gen_ai.request.tools.count", len(kwargs["tools"]))
        tool_names = []
        for tool in kwargs["tools"]:
            if isinstance(tool, dict):
                if "function_declarations" in tool:
                    for func in tool["function_declarations"]:
                        tool_names.append(func.get("name", "unknown"))
                elif "function" in tool:
                    tool_names.append(tool["function"].get("name", "unknown"))
        if tool_names:
            span.set_attribute("gen_ai.request.tools.names", ",".join(tool_names))

    # Safety settings
    if hasattr(llm_instance, "safety_settings") and llm_instance.safety_settings:
        span.set_attribute("gen_ai.request.vertex_ai.safety_settings.enabled", True)


def _set_response_attributes(span, result):
    """Set response-specific attributes following GenAI semantic conventions"""
    if hasattr(result, "generations") and result.generations:
        generation = result.generations[0]
        message = generation.message

        # Usage metadata
        if hasattr(message, "usage_metadata") and message.usage_metadata:
            usage = message.usage_metadata

            if isinstance(usage, dict):
                if "input_tokens" in usage or "prompt_token_count" in usage:
                    input_tokens = usage.get("input_tokens") or usage.get("prompt_token_count")
                    if input_tokens:
                        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                
                if "output_tokens" in usage or "candidates_token_count" in usage:
                    output_tokens = usage.get("output_tokens") or usage.get("candidates_token_count")
                    if output_tokens:
                        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                
                if "total_tokens" in usage or "total_token_count" in usage:
                    total_tokens = usage.get("total_tokens") or usage.get("total_token_count")
                    if total_tokens:
                        span.set_attribute("gen_ai.usage.total_tokens", total_tokens)
            else:
                # Object format
                if hasattr(usage, "prompt_token_count"):
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_token_count)
                if hasattr(usage, "candidates_token_count"):
                    span.set_attribute("gen_ai.usage.output_tokens", usage.candidates_token_count)
                if hasattr(usage, "total_token_count"):
                    span.set_attribute("gen_ai.usage.total_tokens", usage.total_token_count)

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

            if "model_name" in metadata:
                span.set_attribute("gen_ai.response.model", metadata["model_name"])

            # Finish reason
            if "finish_reason" in metadata:
                span.set_attribute("gen_ai.response.finish_reasons", metadata["finish_reason"])

            # Safety ratings
            if "safety_ratings" in metadata and metadata["safety_ratings"]:
                span.set_attribute("gen_ai.response.vertex_ai.safety_ratings.count", 
                                 len(metadata["safety_ratings"]))

            # Citation metadata
            if "citation_metadata" in metadata and metadata["citation_metadata"]:
                span.set_attribute("gen_ai.response.vertex_ai.has_citations", True)


def _wrap_async_methods(llm_instance):
    """Wrap async methods if they exist"""
    if hasattr(llm_instance, "_agenerate"):
        _wrap_agenerate_method(llm_instance)


def _wrap_agenerate_method(llm_instance):
    """Wrap the _agenerate method for async invocations"""
    original_agenerate = llm_instance._agenerate

    @functools.wraps(original_agenerate)
    async def traced_agenerate(messages, stop=None, run_manager=None, **kwargs):
        with spyglass_tracer.start_as_current_span(
            "vertexai.chat.agenerate", record_exception=False
        ) as span:
            try:
                _set_vertexai_attributes(span, llm_instance, messages, kwargs)

                result = await original_agenerate(messages, stop, run_manager, **kwargs)

                _set_response_attributes(span, result)

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    llm_instance._agenerate = traced_agenerate


def _format_langchain_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Format LangChain messages to GenAI semantic convention format."""
    formatted_messages = []

    for message in messages:
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
                text_parts = []
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)

        formatted_message = {"role": role, "content": content}

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            formatted_message["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("args", {})) if tc.get("args") else "",
                    },
                }
                for tc in message.tool_calls
            ]

        # Handle tool call results
        if hasattr(message, "tool_call_id"):
            formatted_message["tool_call_id"] = message.tool_call_id

        formatted_messages.append(formatted_message)

    return formatted_messages

