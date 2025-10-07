import pytest
from unittest.mock import Mock, MagicMock, patch
from opentelemetry.trace import Status, StatusCode

from spyglass_ai.langchain_aws import (
    spyglass_chatbedrockconverse,
    _set_bedrock_attributes,
    _set_response_attributes,
    _format_langchain_messages,
)


class TestSpyglassChatBedrockConverse:
    """Test suite for ChatBedrockConverse tracing integration"""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock ChatBedrockConverse instance"""
        llm = Mock()
        llm.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        llm.provider = "anthropic"
        llm.region_name = "us-west-2"
        llm.temperature = 0.1
        llm.max_tokens = 1000
        llm.top_p = 0.9
        llm.guardrail_config = None
        llm.performance_config = None

        # Mock methods
        llm._generate = Mock()

        return llm

    @pytest.fixture
    def mock_span(self):
        """Create a mock span for testing"""
        span = Mock()
        span.set_attribute = Mock()
        span.set_status = Mock()
        span.record_exception = Mock()
        return span

    @pytest.fixture
    def mock_messages(self):
        """Create mock messages"""
        message1 = Mock(content="Hello, how are you?")
        message1.tool_calls = None
        message2 = Mock(content="I'm doing well, thanks!")
        message2.tool_calls = None
        return [message1, message2]

    def test_spyglass_chatbedrockconverse_wraps_methods(self, mock_llm):
        """Test that spyglass_chatbedrockconverse wraps all expected methods"""
        original_generate = mock_llm._generate

        wrapped_llm = spyglass_chatbedrockconverse(mock_llm)

        # Verify the same instance is returned
        assert wrapped_llm is mock_llm

        # Verify methods are wrapped (different from originals)
        assert mock_llm._generate is not original_generate

    @patch("spyglass_ai.langchain_aws.json.dumps")
    @patch("spyglass_ai.langchain_aws.spyglass_tracer")
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
        wrapped_llm = spyglass_chatbedrockconverse(mock_llm)

        # Call the wrapped method
        result = wrapped_llm._generate(mock_messages)

        # Verify tracing
        mock_tracer.start_as_current_span.assert_called_with(
            "bedrock.chat.generate", record_exception=False
        )
        # Check that set_status was called with OK status (don't check exact object)
        assert mock_span.set_status.called
        status_call = mock_span.set_status.call_args[0][0]
        assert status_call.status_code == StatusCode.OK

        # Verify original method was called (we can't check the wrapped method directly)
        # Instead, verify the result is correct
        assert result is mock_result

    @patch("spyglass_ai.langchain_aws.json.dumps")
    @patch("spyglass_ai.langchain_aws.spyglass_tracer")
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
        wrapped_llm = spyglass_chatbedrockconverse(mock_llm)

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

    @patch("spyglass_ai.langchain_aws.json.dumps")
    def test_set_bedrock_attributes(
        self, mock_json_dumps, mock_span, mock_llm, mock_messages
    ):
        """Test _set_bedrock_attributes function"""
        mock_json_dumps.return_value = '{"messages": "mocked"}'
        kwargs = {
            "tools": [
                {"toolSpec": {"name": "get_weather"}},
                {"function": {"name": "get_population"}},
            ]
        }

        _set_bedrock_attributes(mock_span, mock_llm, mock_messages, kwargs)

        # Verify basic attributes
        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.model", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.aws.provider", "anthropic"
        )
        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.aws.region", "us-west-2"
        )
        mock_span.set_attribute.assert_any_call("gen_ai.request.temperature", 0.1)
        mock_span.set_attribute.assert_any_call("gen_ai.request.max_tokens", 1000)
        mock_span.set_attribute.assert_any_call("gen_ai.request.top_p", 0.9)
        mock_span.set_attribute.assert_any_call("gen_ai.input.messages.count", 2)

        # Verify tool attributes
        mock_span.set_attribute.assert_any_call("gen_ai.request.tools.count", 2)
        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.tools.names", "get_weather,get_population"
        )

    @patch("spyglass_ai.langchain_aws.json.dumps")
    def test_set_bedrock_attributes_with_guardrails(
        self, mock_json_dumps, mock_span, mock_llm, mock_messages
    ):
        """Test _set_bedrock_attributes with guardrails enabled"""
        mock_json_dumps.return_value = '{"messages": "mocked"}'
        mock_llm.guardrail_config = {"guardrailId": "test-guardrail"}
        mock_llm.performance_config = {"latency": "optimized"}

        _set_bedrock_attributes(mock_span, mock_llm, mock_messages, {})

        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.aws.guardrails.enabled", True
        )
        mock_span.set_attribute.assert_any_call(
            "gen_ai.request.aws.performance_config.enabled", True
        )

    @patch("spyglass_ai.langchain_aws.json.dumps")
    def test_set_response_attributes(self, mock_json_dumps, mock_span):
        """Test _set_response_attributes function"""
        mock_json_dumps.return_value = '{"messages": "mocked"}'
        # Create mock response structure
        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.input_token_details = {"cache_read": 10, "cache_creation": 5}

        mock_message = Mock()
        mock_message.usage_metadata = mock_usage
        mock_message.tool_calls = [
            {"name": "get_weather", "id": "call_1"},
            {"name": "get_population", "id": "call_2"},
        ]
        mock_message.response_metadata = {
            "stopReason": "end_turn",
            "metrics": {"latencyMs": 1500},
        }

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_result = Mock()
        mock_result.generations = [mock_generation]

        _set_response_attributes(mock_span, mock_result)

        # Verify usage attributes
        mock_span.set_attribute.assert_any_call("gen_ai.usage.input_tokens", 100)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.output_tokens", 50)
        mock_span.set_attribute.assert_any_call("gen_ai.usage.total_tokens", 150)
        mock_span.set_attribute.assert_any_call(
            "gen_ai.usage.aws.cache_read_tokens", 10
        )
        mock_span.set_attribute.assert_any_call(
            "gen_ai.usage.aws.cache_write_tokens", 5
        )

        # Verify tool call attributes
        mock_span.set_attribute.assert_any_call("gen_ai.response.tools.count", 2)
        mock_span.set_attribute.assert_any_call(
            "gen_ai.response.tools.names", "get_weather,get_population"
        )

        # Verify response metadata
        mock_span.set_attribute.assert_any_call(
            "gen_ai.response.finish_reasons", "end_turn"
        )
        mock_span.set_attribute.assert_any_call("gen_ai.response.aws.latency_ms", 1500)

    @patch("spyglass_ai.langchain_aws.json.dumps")
    def test_latency_handling_list_format(self, mock_json_dumps, mock_span):
        """Test handling of latency in list format"""
        mock_json_dumps.return_value = '{"messages": "mocked"}'
        mock_message = Mock()
        mock_message.usage_metadata = None
        mock_message.tool_calls = None
        mock_message.response_metadata = {
            "metrics": {"latencyMs": [1500]}  # List format
        }

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_result = Mock()
        mock_result.generations = [mock_generation]

        _set_response_attributes(mock_span, mock_result)

        # Should extract first value from list
        mock_span.set_attribute.assert_any_call("gen_ai.response.aws.latency_ms", 1500)
