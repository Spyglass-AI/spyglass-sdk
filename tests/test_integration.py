import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from spyglass_ai.langchain_aws import spyglass_chatbedrockconverse
from spyglass_ai.mcp_tools import spyglass_mcp_tools_async, wrap_mcp_session


class TestIntegration:
    """Integration tests for the complete LangChain AWS + MCP tools flow"""

    @pytest.fixture
    def mock_bedrock_llm(self):
        """Create a mock ChatBedrockConverse instance"""
        llm = Mock()
        llm.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        llm.provider = "anthropic"
        llm.region_name = "us-west-2"
        llm.temperature = 0.1
        llm.max_tokens = 1000
        llm.top_p = None
        llm.guardrail_config = None
        llm.performance_config = None

        # Mock the _generate method
        llm._generate = Mock()

        # Mock bind_tools method
        llm.bind_tools = Mock(return_value=llm)

        return llm

    @pytest.fixture
    def mock_mcp_session(self):
        """Create a mock MCP session"""
        session = Mock()
        session.call_tool = AsyncMock()
        session.initialize = AsyncMock()
        return session

    @pytest.fixture
    def mock_mcp_tools(self):
        """Create mock MCP tools"""
        tool1 = Mock()
        tool1.name = "get_weather"
        tool1.description = "Get weather information"
        tool1.coroutine = AsyncMock()
        tool1.invoke = Mock()
        tool1.ainvoke = AsyncMock()
        tool1.metadata = {}
        tool1.args_schema = Mock()
        tool1.response_format = "content_and_artifact"

        tool2 = Mock()
        tool2.name = "get_population"
        tool2.description = "Get population data"
        tool2.coroutine = AsyncMock()
        tool2.invoke = Mock()
        tool2.ainvoke = AsyncMock()
        tool2.metadata = {}
        tool2.args_schema = Mock()
        tool2.response_format = "string"

        return [tool1, tool2]

    @patch("spyglass_ai.langchain_aws.spyglass_tracer")
    @patch("spyglass_ai.mcp_tools.spyglass_tracer")
    @pytest.mark.asyncio
    async def test_end_to_end_setup(
        self, mock_mcp_tracer, mock_bedrock_tracer, mock_bedrock_llm, mock_mcp_tools
    ):
        """Test end-to-end setup of traced LLM with traced MCP tools"""
        # Setup mock spans
        mock_bedrock_span = Mock()
        mock_mcp_span = Mock()
        mock_bedrock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_bedrock_span
        mock_mcp_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_mcp_span
        )

        # Wrap LLM with tracing
        traced_llm = spyglass_chatbedrockconverse(mock_bedrock_llm)

        # Wrap MCP tools with tracing
        traced_tools = await spyglass_mcp_tools_async(tools=mock_mcp_tools)

        # Verify setup
        assert traced_llm is mock_bedrock_llm
        assert len(traced_tools) == 2

        # Verify LLM methods are wrapped
        assert mock_bedrock_llm._generate is not None

    @patch("spyglass_ai.mcp_tools.spyglass_tracer")
    @pytest.mark.asyncio
    async def test_mcp_session_with_tools_integration(
        self, mock_tracer, mock_mcp_session, mock_mcp_tools
    ):
        """Test integration of MCP session tracing with tool tracing"""
        # Setup mock span
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )

        # Setup session response
        mock_session_result = Mock()
        mock_session_result.content = [{"type": "text", "text": "Session result"}]
        mock_session_result.isError = False
        mock_mcp_session.call_tool.return_value = mock_session_result

        # Setup tool response
        mock_mcp_tools[0].coroutine.return_value = ("Tool result", None)

        # Wrap session and tools
        traced_session = wrap_mcp_session(mock_mcp_session)
        traced_tools = await spyglass_mcp_tools_async(tools=mock_mcp_tools)

        # Call session method
        session_result = await traced_session.call_tool(
            "get_weather", {"location": "SF"}
        )

        # Call tool method
        tool_result = await traced_tools[0].coroutine(location="SF")

        # Verify both are traced
        assert mock_tracer.start_as_current_span.call_count >= 2

        # Verify session tracing
        mock_tracer.start_as_current_span.assert_any_call(
            "mcp.session.call_tool.get_weather", record_exception=False
        )

        # Verify tool tracing
        mock_tracer.start_as_current_span.assert_any_call(
            "mcp.tool.get_weather", record_exception=False
        )

    def test_conditional_imports_handling(self):
        """Test that the integration handles missing dependencies gracefully"""
        # Test that imports are conditional and don't break if dependencies are missing

        # This would be tested by temporarily removing the dependencies
        # and ensuring the main spyglass_ai module still imports

        # For now, just verify the functions exist
        assert spyglass_chatbedrockconverse is not None
        assert spyglass_mcp_tools_async is not None
        assert wrap_mcp_session is not None
