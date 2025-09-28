import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from opentelemetry.trace import Status, StatusCode

from spyglass_ai.mcp_tools import (
    spyglass_mcp_tools,
    spyglass_mcp_tools_async,
    _set_tool_attributes,
    _set_tool_result_attributes,
    wrap_mcp_session,
)


class TestSpyglassMCPTools:
    """Test suite for MCP tools tracing integration"""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock MCP tool"""
        tool = Mock()
        tool.name = "get_weather"
        tool.description = "Get weather information for a location"
        tool.coroutine = AsyncMock()
        tool.func = Mock()
        tool.invoke = Mock()
        tool.ainvoke = AsyncMock()
        tool.metadata = {"_meta": {"version": "1.0"}}
        tool.args_schema = Mock()
        tool.args_schema.model_fields = {"location": Mock(), "units": Mock()}
        tool.response_format = "content_and_artifact"
        return tool

    @pytest.fixture
    def mock_span(self):
        """Create a mock span for testing"""
        span = Mock()
        span.set_attribute = Mock()
        span.set_status = Mock()
        span.record_exception = Mock()
        return span

    @pytest.fixture
    def mock_tools_list(self, mock_tool):
        """Create a list of mock tools"""
        tool2 = Mock()
        tool2.name = "get_population"
        tool2.description = "Get population data"
        tool2.coroutine = AsyncMock()
        tool2.metadata = {}
        tool2.args_schema = Mock()
        tool2.response_format = "string"
        return [mock_tool, tool2]

    def test_spyglass_mcp_tools_with_tools_list(self, mock_tools_list):
        """Test spyglass_mcp_tools with provided tools list"""
        traced_tools = spyglass_mcp_tools(tools=mock_tools_list)

        # Should return the same number of tools
        assert len(traced_tools) == 2

        # Tools should be the same instances (wrapped in place)
        assert traced_tools[0] is mock_tools_list[0]
        assert traced_tools[1] is mock_tools_list[1]

    def test_spyglass_mcp_tools_no_tools_no_session_raises_error(self):
        """Test that providing neither tools nor session raises ValueError"""
        with pytest.raises(
            ValueError, match="Either session or connection must be provided"
        ):
            spyglass_mcp_tools()

    def test_spyglass_mcp_tools_no_tools_with_session_raises_error(self):
        """Test that loading tools without async context raises ValueError"""
        mock_session = Mock()

        with pytest.raises(
            ValueError, match="Loading tools automatically requires async context"
        ):
            spyglass_mcp_tools(session=mock_session)

    @pytest.mark.asyncio
    async def test_spyglass_mcp_tools_async_with_tools(self, mock_tools_list):
        """Test spyglass_mcp_tools_async with provided tools"""
        traced_tools = await spyglass_mcp_tools_async(tools=mock_tools_list)

        assert len(traced_tools) == 2
        assert traced_tools[0] is mock_tools_list[0]

    def test_set_tool_attributes(self, mock_span, mock_tool):
        """Test _set_tool_attributes function"""
        kwargs = {"location": "San Francisco", "units": "celsius"}

        _set_tool_attributes(mock_span, mock_tool, kwargs)

        # Verify basic attributes
        mock_span.set_attribute.assert_any_call("mcp.tool.name", "get_weather")
        mock_span.set_attribute.assert_any_call(
            "mcp.tool.description", "Get weather information for a location"
        )
        mock_span.set_attribute.assert_any_call("mcp.tool.args_count", 2)
        mock_span.set_attribute.assert_any_call("mcp.tool.arg_names", "location,units")

        # Verify metadata attributes
        mock_span.set_attribute.assert_any_call("mcp.tool.has_meta", True)
        mock_span.set_attribute.assert_any_call("mcp.tool.metadata_fields", 1)

        # Verify schema attributes
        mock_span.set_attribute.assert_any_call("mcp.tool.has_schema", True)
        mock_span.set_attribute.assert_any_call("mcp.tool.schema_fields", 2)

        # Verify response format
        mock_span.set_attribute.assert_any_call(
            "mcp.tool.response_format", "content_and_artifact"
        )

    def test_set_tool_attributes_minimal_tool(self, mock_span):
        """Test _set_tool_attributes with minimal tool"""
        minimal_tool = Mock()
        minimal_tool.name = "simple_tool"
        minimal_tool.description = None
        minimal_tool.metadata = None
        minimal_tool.args_schema = None

        _set_tool_attributes(mock_span, minimal_tool, {})

        mock_span.set_attribute.assert_any_call("mcp.tool.name", "simple_tool")
        mock_span.set_attribute.assert_any_call("mcp.tool.description", "")
        mock_span.set_attribute.assert_any_call("mcp.tool.args_count", 0)

    def test_set_tool_result_attributes_tuple_result(self, mock_span):
        """Test _set_tool_result_attributes with tuple result"""
        result = ("Weather is sunny and 72°F", [{"type": "image", "data": "base64..."}])

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_any_call("mcp.tool.result.type", "tuple")
        mock_span.set_attribute.assert_any_call(
            "mcp.tool.result.content_type", "string"
        )
        mock_span.set_attribute.assert_any_call("mcp.tool.result.content_length", 25)
        mock_span.set_attribute.assert_any_call("mcp.tool.result.has_artifacts", True)
        mock_span.set_attribute.assert_any_call("mcp.tool.result.artifacts_count", 1)

    def test_set_tool_result_attributes_string_result(self, mock_span):
        """Test _set_tool_result_attributes with string result"""
        result = "Simple string result"

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_any_call("mcp.tool.result.type", "string")
        mock_span.set_attribute.assert_any_call("mcp.tool.result.content_length", 20)

    def test_set_tool_result_attributes_list_result(self, mock_span):
        """Test _set_tool_result_attributes with list result"""
        result = ["item1", "item2", "item3"]

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_any_call("mcp.tool.result.type", "list")
        mock_span.set_attribute.assert_any_call("mcp.tool.result.content_count", 3)

    def test_set_tool_result_attributes_dict_result(self, mock_span):
        """Test _set_tool_result_attributes with dict result"""
        result = {"temperature": 72, "condition": "sunny", "humidity": 65}

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_any_call("mcp.tool.result.type", "dict")
        mock_span.set_attribute.assert_any_call("mcp.tool.result.fields_count", 3)

    def test_set_tool_result_attributes_none_result(self, mock_span):
        """Test _set_tool_result_attributes with None result"""
        result = None

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_called_once_with("mcp.tool.result.type", "none")

    def test_set_tool_result_attributes_tuple_with_list_content(self, mock_span):
        """Test _set_tool_result_attributes with tuple containing list content"""
        result = (["item1", "item2"], None)

        _set_tool_result_attributes(mock_span, result)

        mock_span.set_attribute.assert_any_call("mcp.tool.result.type", "tuple")
        mock_span.set_attribute.assert_any_call("mcp.tool.result.content_type", "list")
        mock_span.set_attribute.assert_any_call("mcp.tool.result.content_count", 2)
        mock_span.set_attribute.assert_any_call("mcp.tool.result.has_artifacts", False)

    def test_wrap_mcp_session_no_call_tool_method(self):
        """Test wrapping session without call_tool method"""
        mock_session = Mock()
        # Remove call_tool attribute
        if hasattr(mock_session, "call_tool"):
            delattr(mock_session, "call_tool")

        # Should not raise an error
        wrapped_session = wrap_mcp_session(mock_session)
        assert wrapped_session is mock_session
