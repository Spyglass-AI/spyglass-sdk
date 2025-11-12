"""Spyglass integration for pydantic-ai agents."""

try:
    from pydantic_ai import Agent

    _PYDANTIC_AI_AVAILABLE = True
except ImportError:
    _PYDANTIC_AI_AVAILABLE = False

from .otel import get_spyglass_tracer


def spyglass_pydantic(agent_instance):
    """
    Wraps a pydantic-ai Agent instance to add comprehensive tracing.

    This wrapper enables OpenTelemetry instrumentation for the entire agent flow,
    including agent runs, model requests, tool calls, and output validation.
    It follows OpenTelemetry GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    Args:
        agent_instance: A pydantic-ai Agent instance

    Returns:
        The same agent instance with instrumentation enabled

    Raises:
        ImportError: If pydantic-ai is not installed

    Example:
        ```python
        from pydantic_ai import Agent
        from spyglass_ai import spyglass_pydantic

        agent = Agent('openai:gpt-4o')
        agent = spyglass_pydantic(agent)

        # Now all agent runs will be traced
        result = agent.run_sync('What is the capital of France?')
        ```
    """
    if not _PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is required for spyglass_pydantic. "
            "Install with: pip install pydantic-ai"
        )

    if not isinstance(agent_instance, Agent):
        raise TypeError(
            f"spyglass_pydantic expects a pydantic_ai.Agent instance, "
            f"got {type(agent_instance).__name__}"
        )

    # Ensure spyglass tracer provider is initialized
    # This sets up the global tracer provider that pydantic-ai will use
    get_spyglass_tracer()

    # Enable instrumentation on the agent
    # Setting instrument=True will use the global tracer provider
    # that spyglass just initialized, which provides automatic tracing
    # of agent runs, model requests, tool calls, and output validation
    agent_instance.instrument = True

    return agent_instance
