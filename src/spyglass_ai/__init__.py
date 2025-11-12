from .openai import spyglass_openai
from .otel import configure_spyglass
from .trace import spyglass_trace

# LangChain AWS integrations
try:
    from .langchain_aws import spyglass_chatbedrockconverse

    _LANGCHAIN_AWS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AWS_AVAILABLE = False

# LangChain OpenAI integrations
try:
    from .langchain_openai import spyglass_chatopenai

    _LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_OPENAI_AVAILABLE = False

# MCP tools integrations
try:
    from .mcp_tools import (
        spyglass_mcp_tools,
        spyglass_mcp_tools_async,
        wrap_mcp_session,
    )

    _MCP_TOOLS_AVAILABLE = True
except ImportError:
    _MCP_TOOLS_AVAILABLE = False

# Pydantic AI integrations
try:
    from .pydantic import spyglass_pydantic

    _PYDANTIC_AI_AVAILABLE = True
except ImportError:
    _PYDANTIC_AI_AVAILABLE = False

# Base exports
__all__ = ["spyglass_trace", "spyglass_openai", "configure_spyglass"]

# Add conditional exports
if _LANGCHAIN_AWS_AVAILABLE:
    __all__.append("spyglass_chatbedrockconverse")

if _LANGCHAIN_OPENAI_AVAILABLE:
    __all__.append("spyglass_chatopenai")

if _MCP_TOOLS_AVAILABLE:
    __all__.extend(
        ["spyglass_mcp_tools", "spyglass_mcp_tools_async", "wrap_mcp_session"]
    )

if _PYDANTIC_AI_AVAILABLE:
    __all__.append("spyglass_pydantic")
