from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.trace import StatusCode

from spyglass_ai.openai import spyglass_openai


class MockOpenAIResponse:
    """Mock OpenAI API response object."""

    def __init__(self, model="gpt-3.5-turbo", usage=None):
        self.model = model
        self.usage = usage


class MockUsage:
    """Mock usage object from OpenAI response."""

    def __init__(self, prompt_tokens=10, completion_tokens=20, total_tokens=30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock()


class TestSpyglassOpenAI:
    """Test the spyglass_openai wrapper function."""

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_basic_wrapping(self, mock_tracer):
        """Test that the client is properly wrapped."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()
        original_create = client.chat.completions.create

        wrapped_client = spyglass_openai(client)

        # Should return the same client instance
        assert wrapped_client is client

        # The create method should be replaced
        assert client.chat.completions.create != original_create

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_span_creation(self, mock_tracer):
        """Test that spans are created with correct names."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()
        client.chat.completions.create.return_value = MockOpenAIResponse()

        wrapped_client = spyglass_openai(client)

        # Call the wrapped method
        wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
        )

        # Check span was created with correct name
        mock_tracer.start_as_current_span.assert_called_once_with(
            "openai.chat.completions.create", record_exception=False
        )

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_request_attributes(self, mock_tracer):
        """Test that request attributes are properly captured."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()
        client.chat.completions.create.return_value = MockOpenAIResponse()

        wrapped_client = spyglass_openai(client)

        # Call with various parameters
        wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        # Check that attributes were set
        expected_calls = [
            ("gen_ai.request.model", "gpt-4"),
            ("gen_ai.input.messages.count", 2),
            ("gen_ai.request.max_tokens", 100),
            ("gen_ai.request.temperature", 0.7),
        ]

        for expected_call in expected_calls:
            mock_span.set_attribute.assert_any_call(*expected_call)

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_response_attributes(self, mock_tracer):
        """Test that response attributes are properly captured."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()

        # Create a response with usage information
        usage = MockUsage(prompt_tokens=15, completion_tokens=25, total_tokens=40)
        response = MockOpenAIResponse(model="gpt-4", usage=usage)
        client.chat.completions.create.return_value = response

        wrapped_client = spyglass_openai(client)

        wrapped_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )

        # Check that response attributes were set
        expected_calls = [
            ("gen_ai.usage.input_tokens", 15),
            ("gen_ai.usage.output_tokens", 25),
            ("gen_ai.usage.total_tokens", 40),
            ("gen_ai.response.model", "gpt-4"),
        ]

        for expected_call in expected_calls:
            mock_span.set_attribute.assert_any_call(*expected_call)

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_exception_handling(self, mock_tracer):
        """Test exception handling in the wrapper."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()

        # Make the original method raise an exception
        test_exception = Exception("API Error")
        client.chat.completions.create.side_effect = test_exception

        wrapped_client = spyglass_openai(client)

        # Call should raise the original exception
        with pytest.raises(Exception, match="API Error"):
            wrapped_client.chat.completions.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
            )

        # Check that exception was recorded
        mock_span.record_exception.assert_called_once_with(test_exception)
        # Check status was set to ERROR with correct message
        call_args = mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "API Error"

    @patch("spyglass_ai.openai.spyglass_tracer")
    def test_return_value(self, mock_tracer):
        """Test that the original return value is preserved."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        client = MockOpenAIClient()
        expected_response = MockOpenAIResponse(model="gpt-4")
        client.chat.completions.create.return_value = expected_response

        wrapped_client = spyglass_openai(client)

        result = wrapped_client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )

        # Should return the original response
        assert result is expected_response


class TestIntegrationWithRealClient:
    """Integration tests that could work with a real OpenAI client."""

    # TODO: Implement this test with a real OpenAI client and API key read from env
    @pytest.mark.skip(reason="Requires real OpenAI client and API key")
    def test_real_openai_client_integration(self):
        """Test with a real OpenAI client (skipped by default)."""
        # This test would require:
        # import openai
        # client = openai.OpenAI(api_key="your-api-key")
        # wrapped_client = spyglass_openai(client)
        # response = wrapped_client.chat.completions.create(...)
        pass
