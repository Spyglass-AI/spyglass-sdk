# LangChain AWS + MCP Tools Integration

Comprehensive tracing integration for LangChain AWS Bedrock models and MCP (Model Context Protocol) tools in the Spyglass SDK.

## Overview

The Spyglass SDK supports tracing:
- **ChatBedrockConverse**: AWS Bedrock chat models via LangChain
- **MCP Tools**: Model Context Protocol tools and sessions
- **Complete Integration**: End-to-end tracing from LLM calls through tool executions

## Installation

### Base Installation
```bash
pip install spyglass-ai
```

### With LangChain AWS Support
```bash
pip install spyglass-ai[langchain-aws]
```

### With MCP Tools Support
```bash
pip install spyglass-ai[mcp]
```

### Full Installation
```bash
pip install spyglass-ai[langchain-aws,mcp]
```

**Requirements:**
- Python 3.10+ (required by MCP adapters)
- AWS credentials configured
- Spyglass deployment ID and API key

## Quick Start

### 1. Basic ChatBedrockConverse Tracing

```python
from spyglass_ai import spyglass_chatbedrockconverse
from langchain_aws import ChatBedrockConverse

# Create your LLM
llm = ChatBedrockConverse(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-west-2",
    temperature=0.1,
)

# Wrap with Spyglass tracing
traced_llm = spyglass_chatbedrockconverse(llm)

# Use normally - all calls are now traced
response = traced_llm.invoke("Hello, how are you?")
```

### 2. MCP Tools Tracing

```python
from spyglass_ai import spyglass_mcp_tools_async, wrap_mcp_session
from langchain_mcp_adapters.sessions import create_session, Connection

# Setup MCP connection
connection = Connection(command="python", args=["-m", "your_mcp_server"])

async with create_session(connection) as session:
    await session.initialize()
    
    # Wrap session with tracing
    traced_session = wrap_mcp_session(session)
    
    # Load and trace tools
    traced_tools = await spyglass_mcp_tools_async(session=traced_session)
    
    # All tool executions are now traced
    result = await traced_tools[0].ainvoke({"param": "value"})
```

### 3. Complete Integration

```python
from spyglass_ai import spyglass_chatbedrockconverse, spyglass_mcp_tools_async
from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.sessions import create_session, Connection

# Setup traced LLM
llm = ChatBedrockConverse(model="anthropic.claude-3-sonnet-20240229-v1:0")
traced_llm = spyglass_chatbedrockconverse(llm)

# Setup traced MCP tools
connection = Connection(command="python", args=["-m", "your_mcp_server"])
async with create_session(connection) as session:
    await session.initialize()
    traced_tools = await spyglass_mcp_tools_async(session=session)
    
    # Bind traced tools to traced LLM
    llm_with_tools = traced_llm.bind_tools(traced_tools)
    
    # Complete end-to-end tracing
    response = await llm_with_tools.ainvoke("Help me with a task")
```



## Configuration

### Environment Variables

Required:
```bash
export SPYGLASS_DEPLOYMENT_ID="your-deployment-id"
export SPYGLASS_API_KEY="your-api-key"
```

### AWS Configuration

Standard AWS configuration applies:
```bash
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
```

## Traced Spans and Attributes

### Span Names

The integration creates spans with descriptive names:

- `bedrock.chat.generate` - Sync LLM generation
- `bedrock.chat.agenerate` - Async LLM generation
- `mcp.tool.{tool_name}` - MCP tool execution
- `mcp.tool.{tool_name}.invoke` - Tool invoke method
- `mcp.tool.{tool_name}.ainvoke` - Tool async invoke method
- `mcp.session.call_tool.{tool_name}` - Session-level tool call

### ChatBedrockConverse Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.system` | AI system | `aws_bedrock` |
| `gen_ai.request.model` | Model identifier | `anthropic.claude-3-sonnet-20240229-v1:0` |
| `gen_ai.request.aws.provider` | Model provider | `anthropic` |
| `gen_ai.request.aws.region` | AWS region | `us-west-2` |
| `gen_ai.request.temperature` | Sampling temperature | `0.1` |
| `gen_ai.request.max_tokens` | Maximum tokens | `1000` |
| `gen_ai.input.messages.count` | Number of input messages | `3` |
| `gen_ai.request.tools.count` | Number of available tools | `5` |
| `gen_ai.request.tools.names` | Tool names (comma-separated) | `get_weather,get_population` |
| `gen_ai.usage.input_tokens` | Input tokens consumed | `150` |
| `gen_ai.usage.output_tokens` | Output tokens generated | `75` |
| `gen_ai.usage.total_tokens` | Total tokens | `225` |
| `gen_ai.usage.aws.cache_read_tokens` | Cache read tokens | `50` |
| `gen_ai.response.tools.count` | Number of tool calls | `2` |
| `gen_ai.response.finish_reasons` | Why generation stopped | `end_turn` |
| `gen_ai.response.aws.latency_ms` | Response latency | `1500` |
| `gen_ai.request.aws.guardrails.enabled` | Guardrails active | `true` |

### MCP Tools Attributes

| Attribute | Description | Example |
|-----------|-------------|---------|
| `mcp.tool.name` | Tool name | `get_weather` |
| `mcp.tool.description` | Tool description | `Get weather information` |
| `mcp.tool.args_count` | Number of arguments | `2` |
| `mcp.tool.arg_names` | Argument names | `location,units` |
| `mcp.tool.has_schema` | Has input schema | `true` |
| `mcp.tool.response_format` | Response format | `content_and_artifact` |
| `mcp.tool.result.type` | Result type | `tuple` |
| `mcp.tool.result.content_length` | Content length | `150` |
| `mcp.tool.result.has_artifacts` | Has artifacts | `true` |
| `mcp.session.tool_name` | Session-level tool name | `get_weather` |
| `mcp.session.result.is_error` | Tool returned error | `false` |

## Error Handling

All errors are automatically captured and traced:

```python
try:
    response = traced_llm.invoke("This might fail")
except Exception as e:
    # Error is automatically recorded in the trace
    # with full exception details and stack trace
    pass
```

Error information includes:
- Exception type and message
- Full stack trace
- Span status set to ERROR
- Error recorded as span event

## Performance Considerations

The tracing integration is designed to have minimal performance impact:

- **Lazy Attribute Setting**: Attributes are only set when values exist
- **Efficient Wrapping**: Uses `functools.wraps` to preserve function metadata
- **Minimal Overhead**: Tracing operations are lightweight
- **Async Support**: Full support for async operations without blocking
- **Session Reuse**: Supports persistent MCP sessions for better performance

## Advanced Usage

### Custom Tool Filtering

```python
# Trace only specific tools
specific_tools = [tool for tool in all_tools if tool.name in ["get_weather", "get_time"]]
traced_tools = await spyglass_mcp_tools_async(tools=specific_tools)
```

### Session Reuse

```python
# Reuse MCP session for better performance
async with create_session(connection) as session:
    await session.initialize()
    traced_session = wrap_mcp_session(session)
    
    # Multiple tool loads with same session
    tools1 = await spyglass_mcp_tools_async(session=traced_session)
    tools2 = await spyglass_mcp_tools_async(session=traced_session)
```

## Examples

See `examples/langchain_aws_mcp_example.py` for complete working examples including:
- Basic Bedrock tracing
- MCP tools integration
- Error handling
- Performance patterns

## Support

For issues specific to this integration:
1. Check the [GitHub Issues](https://github.com/spyglass-ai/spyglass-sdk/issues)
2. Review the example code
3. Verify your environment configuration
4. Contact support with trace IDs for specific issues
