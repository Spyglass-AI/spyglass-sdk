import pytest
from unittest.mock import Mock, AsyncMock, patch
from opentelemetry.trace import Status, StatusCode

from spyglass_ai.langchain_openai import spyglass_chatopenai


class TestSpyglassChatOpenAI:
    """Test suite for ChatOpenAI tracing integration"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock ChatOpenAI instance"""
        llm = Mock()
        llm.model_name = "gpt-4"
        llm.temperature = 0.7
        llm.max_tokens = 1000
        llm.top_p = 0.9

        # Mock methods
        llm._generate = Mock()
        llm._agenerate = AsyncMock()

        return llm

    @pytest.fixture
    def mock_messages(self):
        """Create mock messages"""
        message1 = Mock()
        message1.__class__.__name__ = "HumanMessage"
        message1.content = "Hello, how are you?"
        message1.tool_calls = None
        return [message1]

    def test_spyglass_chatopenai_wraps_methods(self, mock_llm):
        """Test that spyglass_chatopenai wraps all expected methods"""
        original_generate = mock_llm._generate
        original_agenerate = mock_llm._agenerate

        wrapped_llm = spyglass_chatopenai(mock_llm)

        # Verify the same instance is returned
        assert wrapped_llm is mock_llm

        # Verify methods are wrapped (different from originals)
        assert mock_llm._generate is not original_generate
        assert mock_llm._agenerate is not original_agenerate

    @patch("spyglass_ai.langchain_openai.json.dumps")
    @patch("spyglass_ai.langchain_openai.spyglass_tracer")
    def test_generate_method_tracing(
        self, mock_tracer, mock_json_dumps, mock_llm, mock_messages
    ):
        """Test that _generate method is properly traced"""
        # Setup
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_json_dumps.return_value = '{"messages": "mocked"}'

        # Create a proper mock result structure
        mock_generation = Mock()
        mock_generation.message = Mock()
        mock_generation.message.usage_metadata = None
        mock_generation.message.tool_calls = None
        mock_generation.message.response_metadata = {}

        mock_result = Mock()
        mock_result.generations = [mock_generation]
        mock_llm._generate.return_value = mock_result

        # Wrap the LLM
        wrapped_llm = spyglass_chatopenai(mock_llm)

        # Call the wrapped method
        result = wrapped_llm._generate(mock_messages)

        # Verify tracing
        mock_tracer.start_as_current_span.assert_called_with(
            "openai.chat.generate", record_exception=False
        )
        # Check that set_status was called with OK status
        assert mock_span.set_status.called
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK

        # Verify original method was called
        assert result is mock_result

    @patch("spyglass_ai.langchain_openai.json.dumps")
    @patch("spyglass_ai.langchain_openai.spyglass_tracer")
    def test_generate_method_exception_handling(
        self, mock_tracer, mock_json_dumps, mock_llm, mock_messages
    ):
        """Test exception handling in _generate method"""
        # Setup
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_json_dumps.return_value = '{"messages": "mocked"}'

        test_exception = Exception("Test error")
        mock_llm._generate.side_effect = test_exception

        # Wrap the LLM
        wrapped_llm = spyglass_chatopenai(mock_llm)

        # Call the wrapped method and expect exception
        with pytest.raises(Exception, match="Test error"):
            wrapped_llm._generate(mock_messages)

        # Verify exception was recorded
        mock_span.record_exception.assert_called_once_with(test_exception)
        # Check that set_status was called with ERROR status
        assert mock_span.set_status.called
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.ERROR
        assert "Test error" in str(status_call.description)

    @patch("spyglass_ai.langchain_openai.json.dumps")
    @patch("spyglass_ai.langchain_openai.spyglass_tracer")
    @pytest.mark.asyncio
    async def test_agenerate_method_tracing(
        self, mock_tracer, mock_json_dumps, mock_llm, mock_messages
    ):
        """Test that _agenerate method is properly traced"""
        # Setup
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_json_dumps.return_value = '{"messages": "mocked"}'

        # Create a proper mock result structure
        mock_generation = Mock()
        mock_generation.message = Mock()
        mock_generation.message.usage_metadata = None
        mock_generation.message.tool_calls = None
        mock_generation.message.response_metadata = {}

        mock_result = Mock()
        mock_result.generations = [mock_generation]
        mock_llm._agenerate.return_value = mock_result

        # Wrap the LLM
        wrapped_llm = spyglass_chatopenai(mock_llm)

        # Call the wrapped async method
        result = await wrapped_llm._agenerate(mock_messages)

        # Verify tracing
        mock_tracer.start_as_current_span.assert_called_with(
            "openai.chat.agenerate", record_exception=False
        )
        # Check that set_status was called with OK status
        assert mock_span.set_status.called
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK

        # Verify original method was called
        assert result is mock_result

