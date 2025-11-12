import functools
from typing import Any, List, Optional, Union

from opentelemetry.trace import Status, StatusCode

from .otel import spyglass_tracer


def spyglass_mcp_tools(session=None, tools=None, connection=None):
    """
    Wraps MCP tools with tracing capabilities.

    Args:
        session: MCP client session (optional)
        tools: Optional list of specific tools to wrap
        connection: Optional connection config for creating session on-demand

    Returns:
        List of traced LangChain tools
    """
    if tools is None:
        # Import here to avoid circular dependencies
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ImportError:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP tools tracing. "
                "Install with: pip install langchain-mcp-adapters"
            )

        # Load all tools if none specified
        if session is None and connection is None:
            raise ValueError(
                "Either session or connection must be provided when tools is None"
            )

        # This would need to be called in an async context
        raise ValueError(
            "Loading tools automatically requires async context. "
            "Please load tools first and pass them to this function."
        )

    traced_tools = []
    for tool in tools:
        traced_tool = _wrap_mcp_tool(tool)
        traced_tools.append(traced_tool)

    return traced_tools


async def spyglass_mcp_tools_async(session=None, tools=None, connection=None):
    """
    Async version that can load and wrap MCP tools with tracing capabilities.

    Args:
        session: MCP client session (optional)
        tools: Optional list of specific tools to wrap
        connection: Optional connection config for creating session on-demand

    Returns:
        List of traced LangChain tools
    """
    if tools is None:
        # Import here to avoid circular dependencies
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ImportError:
            raise ImportError(
                "langchain-mcp-adapters is required for MCP tools tracing. "
                "Install with: pip install langchain-mcp-adapters"
            )

        # Load all tools if none specified
        tools = await load_mcp_tools(session, connection=connection)

    traced_tools = []
    for tool in tools:
        traced_tool = _wrap_mcp_tool(tool)
        traced_tools.append(traced_tool)

    return traced_tools


def _wrap_mcp_tool(tool):
    """Wrap individual MCP tool with tracing"""
    # For StructuredTool, we need to create a new instance since we can't modify Pydantic models
    if hasattr(tool, "__class__") and tool.__class__.__name__ == "StructuredTool":
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            # Fall back to original tool if import fails
            return tool

        # Store original coroutine if it exists
        original_coroutine = getattr(tool, "coroutine", None)
        original_func = getattr(tool, "func", None)

        traced_coroutine = None
        traced_func = None

        if original_coroutine:

            async def traced_coroutine(**kwargs):
                with spyglass_tracer.start_as_current_span(
                    f"mcp.tool.{tool.name}", record_exception=False
                ) as span:
                    try:
                        # Set tool attributes
                        _set_tool_attributes(span, tool, kwargs)

                        # Execute tool
                        result = await original_coroutine(**kwargs)

                        # Set result attributes
                        _set_tool_result_attributes(span, result)

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            # Safely wrap function metadata, handling union type annotations
            # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
            # Instead, manually copy safe attributes
            try:
                traced_coroutine.__name__ = getattr(
                    original_coroutine, "__name__", "traced_coroutine"
                )
                traced_coroutine.__doc__ = getattr(original_coroutine, "__doc__", None)
                traced_coroutine.__module__ = getattr(
                    original_coroutine, "__module__", None
                )
            except (TypeError, AttributeError):
                pass  # If copying fails, continue without metadata

        if original_func:

            def traced_func(**kwargs):
                with spyglass_tracer.start_as_current_span(
                    f"mcp.tool.{tool.name}", record_exception=False
                ) as span:
                    try:
                        # Set tool attributes
                        _set_tool_attributes(span, tool, kwargs)

                        # Execute tool
                        result = original_func(**kwargs)

                        # Set result attributes
                        _set_tool_result_attributes(span, result)

                        span.set_status(Status(StatusCode.OK))
                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            # Safely wrap function metadata, handling union type annotations
            # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
            # Instead, manually copy safe attributes
            try:
                traced_func.__name__ = getattr(original_func, "__name__", "traced_func")
                traced_func.__doc__ = getattr(original_func, "__doc__", None)
                traced_func.__module__ = getattr(original_func, "__module__", None)
            except (TypeError, AttributeError):
                pass  # If copying fails, continue without metadata

        # Create new StructuredTool with traced functions
        try:
            new_tool = StructuredTool(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
                func=traced_func,
                coroutine=traced_coroutine,
                response_format=getattr(tool, "response_format", "content"),
                metadata=getattr(tool, "metadata", None),
            )
            
            # Also wrap ainvoke method for StructuredTool since LangChain calls it directly
            if hasattr(new_tool, "ainvoke"):
                original_ainvoke = new_tool.ainvoke
                
                async def traced_ainvoke(self, input_data, config=None, **kwargs):
                    with spyglass_tracer.start_as_current_span(
                        f"mcp.tool.{tool.name}.ainvoke", record_exception=False
                    ) as span:
                        try:
                            # Set tool attributes
                            _set_tool_attributes(span, self, {"input": input_data})
                            
                            # Execute tool - call the original method with self bound
                            result = await original_ainvoke(input_data, config, **kwargs)
                            
                            # Set result attributes
                            _set_tool_result_attributes(span, result)
                            
                            span.set_status(Status(StatusCode.OK))
                            return result
                            
                        except Exception as e:
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise
                
                # Replace ainvoke method using __dict__ since StructuredTool is a Pydantic model
                # Bind the traced function to the tool instance
                try:
                    import types
                    bound_traced_ainvoke = types.MethodType(traced_ainvoke, new_tool)
                    new_tool.__dict__["ainvoke"] = bound_traced_ainvoke
                except (AttributeError, ValueError, TypeError):
                    # If we can't replace it, the coroutine wrapping should still work
                    pass
            
            return new_tool
        except Exception:
            # If creating new tool fails, return original
            return tool

    # For other tool types, try the original approach
    # Store original coroutine if it exists
    original_coroutine = getattr(tool, "coroutine", None)
    original_func = getattr(tool, "func", None)

    if original_coroutine:

        async def traced_coroutine(**kwargs):
            with spyglass_tracer.start_as_current_span(
                f"mcp.tool.{tool.name}", record_exception=False
            ) as span:
                try:
                    # Set tool attributes
                    _set_tool_attributes(span, tool, kwargs)

                    # Execute tool
                    result = await original_coroutine(**kwargs)

                    # Set result attributes
                    _set_tool_result_attributes(span, result)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Safely wrap function metadata, handling union type annotations
        # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
        # Instead, manually copy safe attributes
        try:
            traced_coroutine.__name__ = getattr(
                original_coroutine, "__name__", "traced_coroutine"
            )
            traced_coroutine.__doc__ = getattr(original_coroutine, "__doc__", None)
            traced_coroutine.__module__ = getattr(
                original_coroutine, "__module__", None
            )
        except (TypeError, AttributeError):
            pass  # If copying fails, continue without metadata

        # Try to replace the coroutine with our traced version
        try:
            tool.coroutine = traced_coroutine
        except (AttributeError, ValueError):
            # Can't modify this tool type, continue with other methods
            pass

    elif original_func:

        def traced_func(**kwargs):
            with spyglass_tracer.start_as_current_span(
                f"mcp.tool.{tool.name}", record_exception=False
            ) as span:
                try:
                    # Set tool attributes
                    _set_tool_attributes(span, tool, kwargs)

                    # Execute tool
                    result = original_func(**kwargs)

                    # Set result attributes
                    _set_tool_result_attributes(span, result)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Safely wrap function metadata, handling union type annotations
        # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
        # Instead, manually copy safe attributes
        try:
            traced_func.__name__ = getattr(original_func, "__name__", "traced_func")
            traced_func.__doc__ = getattr(original_func, "__doc__", None)
            traced_func.__module__ = getattr(original_func, "__module__", None)
        except (TypeError, AttributeError):
            pass  # If copying fails, continue without metadata

        # Try to replace the function with our traced version
        try:
            tool.func = traced_func
        except (AttributeError, ValueError):
            # Can't modify this tool type, continue with other methods
            pass

    # Also wrap the invoke methods if they exist
    if hasattr(tool, "invoke"):
        original_invoke = tool.invoke

        def traced_invoke(input_data, config=None, **kwargs):
            with spyglass_tracer.start_as_current_span(
                f"mcp.tool.{tool.name}.invoke", record_exception=False
            ) as span:
                try:
                    # Set tool attributes
                    _set_tool_attributes(span, tool, {"input": input_data})

                    # Execute tool
                    result = original_invoke(input_data, config, **kwargs)

                    # Set result attributes
                    _set_tool_result_attributes(span, result)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Safely wrap function metadata, handling union type annotations
        # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
        # Instead, manually copy safe attributes
        try:
            traced_invoke.__name__ = getattr(
                original_invoke, "__name__", "traced_invoke"
            )
            traced_invoke.__doc__ = getattr(original_invoke, "__doc__", None)
            traced_invoke.__module__ = getattr(original_invoke, "__module__", None)
        except (TypeError, AttributeError):
            pass

        try:
            tool.invoke = traced_invoke
        except (AttributeError, ValueError):
            # Can't modify this tool type, continue with other methods
            pass

    if hasattr(tool, "ainvoke"):
        original_ainvoke = tool.ainvoke

        async def traced_ainvoke(input_data, config=None, **kwargs):
            with spyglass_tracer.start_as_current_span(
                f"mcp.tool.{tool.name}.ainvoke", record_exception=False
            ) as span:
                try:
                    # Set tool attributes
                    _set_tool_attributes(span, tool, {"input": input_data})

                    # Execute tool
                    result = await original_ainvoke(input_data, config, **kwargs)

                    # Set result attributes
                    _set_tool_result_attributes(span, result)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Safely wrap function metadata, handling union type annotations
        # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
        # Instead, manually copy safe attributes
        try:
            traced_ainvoke.__name__ = getattr(
                original_ainvoke, "__name__", "traced_ainvoke"
            )
            traced_ainvoke.__doc__ = getattr(original_ainvoke, "__doc__", None)
            traced_ainvoke.__module__ = getattr(original_ainvoke, "__module__", None)
        except (TypeError, AttributeError):
            pass

        try:
            tool.ainvoke = traced_ainvoke
        except (AttributeError, ValueError):
            # Can't modify this tool type, continue with other methods
            pass

    return tool


def _set_tool_attributes(span, tool, kwargs):
    """Set MCP tool-specific attributes"""
    span.set_attribute("mcp.tool.name", tool.name)
    span.set_attribute("mcp.tool.description", tool.description or "")

    # Count arguments
    if isinstance(kwargs, dict):
        span.set_attribute("mcp.tool.args_count", len(kwargs))

        # Set argument names (but not values for privacy)
        if kwargs:
            arg_names = list(kwargs.keys())
            span.set_attribute("mcp.tool.arg_names", ",".join(arg_names))

    # Tool metadata if available
    if hasattr(tool, "metadata") and tool.metadata:
        metadata = tool.metadata
        if isinstance(metadata, dict):
            # Add selected metadata fields
            if "_meta" in metadata:
                span.set_attribute("mcp.tool.has_meta", True)

            # Count metadata fields
            span.set_attribute("mcp.tool.metadata_fields", len(metadata))

    # Schema information
    if hasattr(tool, "args_schema") and tool.args_schema:
        span.set_attribute("mcp.tool.has_schema", True)
        # Get schema field count if possible
        try:
            if hasattr(tool.args_schema, "model_fields"):
                span.set_attribute(
                    "mcp.tool.schema_fields", len(tool.args_schema.model_fields)
                )
        except Exception:
            pass  # Ignore schema introspection errors

    # Response format if available
    if hasattr(tool, "response_format"):
        span.set_attribute("mcp.tool.response_format", tool.response_format)


def _set_tool_result_attributes(span, result):
    """Set attributes based on tool execution result"""
    if result is None:
        span.set_attribute("mcp.tool.result.type", "none")
        return

    # Handle tuple results (content, artifacts)
    if isinstance(result, tuple) and len(result) == 2:
        content, artifacts = result
        span.set_attribute("mcp.tool.result.type", "tuple")

        # Content analysis
        if isinstance(content, str):
            span.set_attribute("mcp.tool.result.content_type", "string")
            span.set_attribute("mcp.tool.result.content_length", len(content))
        elif isinstance(content, list):
            span.set_attribute("mcp.tool.result.content_type", "list")
            span.set_attribute("mcp.tool.result.content_count", len(content))
        elif content is not None:
            span.set_attribute("mcp.tool.result.content_type", type(content).__name__)

        # Artifacts analysis
        if artifacts is not None:
            span.set_attribute("mcp.tool.result.has_artifacts", True)
            if isinstance(artifacts, list):
                span.set_attribute("mcp.tool.result.artifacts_count", len(artifacts))
        else:
            span.set_attribute("mcp.tool.result.has_artifacts", False)

    # Handle string results
    elif isinstance(result, str):
        span.set_attribute("mcp.tool.result.type", "string")
        span.set_attribute("mcp.tool.result.content_length", len(result))

    # Handle list results
    elif isinstance(result, list):
        span.set_attribute("mcp.tool.result.type", "list")
        span.set_attribute("mcp.tool.result.content_count", len(result))

    # Handle dict results
    elif isinstance(result, dict):
        span.set_attribute("mcp.tool.result.type", "dict")
        span.set_attribute("mcp.tool.result.fields_count", len(result))

    # Handle other types
    else:
        span.set_attribute("mcp.tool.result.type", type(result).__name__)


def wrap_mcp_session(session):
    """
    Wrap an MCP session to trace tool calls at the session level.

    Args:
        session: MCP ClientSession instance

    Returns:
        The same session with tracing enabled
    """
    if hasattr(session, "call_tool"):
        original_call_tool = session.call_tool

        async def traced_call_tool(name, arguments=None):
            with spyglass_tracer.start_as_current_span(
                f"mcp.session.call_tool.{name}", record_exception=False
            ) as span:
                try:
                    # Set session-level attributes
                    span.set_attribute("mcp.session.tool_name", name)
                    if arguments:
                        span.set_attribute("mcp.session.args_count", len(arguments))

                    # Call original method
                    result = await original_call_tool(name, arguments)

                    # Set result attributes
                    if hasattr(result, "content") and result.content:
                        span.set_attribute(
                            "mcp.session.result.content_blocks", len(result.content)
                        )

                    if hasattr(result, "isError"):
                        span.set_attribute(
                            "mcp.session.result.is_error", result.isError
                        )
                        if result.isError:
                            span.set_status(
                                Status(StatusCode.ERROR, "MCP tool returned error")
                            )
                        else:
                            span.set_status(Status(StatusCode.OK))
                    else:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Safely wrap function metadata, handling union type annotations
        # Skip functools.wraps() entirely to avoid UnionType issues in Python 3.10+
        # Instead, manually copy safe attributes
        try:
            traced_call_tool.__name__ = getattr(
                original_call_tool, "__name__", "traced_call_tool"
            )
            traced_call_tool.__doc__ = getattr(original_call_tool, "__doc__", None)
            traced_call_tool.__module__ = getattr(
                original_call_tool, "__module__", None
            )
        except (TypeError, AttributeError):
            pass

        session.call_tool = traced_call_tool

    return session
