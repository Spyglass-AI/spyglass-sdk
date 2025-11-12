import pytest
from unittest.mock import Mock, patch, MagicMock
from spyglass_ai.langchain_google_vertexai import (
    spyglass_chatvertexai,
    _format_langchain_messages,
)


@pytest.fixture
def mock_vertexai_llm():
    """Create a mock ChatVertexAI instance"""
    llm = Mock()
    llm.model_name = "gemini-2.0-flash-exp"
    llm.project = "test-project"
    llm.location = "us-central1"
    llm.temperature = 0.7
    llm.max_output_tokens = 1024
    llm.top_p = 0.95
    llm.top_k = 40
    llm.safety_settings = None
    
    # Mock the _generate method
    llm._generate = Mock()
    
    return llm


@pytest.fixture
def mock_message():
    """Create a mock LangChain message"""
    message = Mock(spec=['content', 'tool_calls', 'usage_metadata', 'response_metadata', '__class__'])
    message.__class__.__name__ = "AIMessage"
    message.content = "This is a test response"
    message.usage_metadata = {
        "prompt_token_count": 10,
        "candidates_token_count": 20,
        "total_token_count": 30
    }
    message.response_metadata = {
        "model_name": "gemini-2.0-flash-exp",
        "finish_reason": "STOP"
    }
    message.tool_calls = []
    return message


@pytest.fixture
def mock_generation(mock_message):
    """Create a mock generation result"""
    generation = Mock()
    generation.message = mock_message
    
    result = Mock()
    result.generations = [generation]
    
    return result


def test_spyglass_chatvertexai_wraps_instance(mock_vertexai_llm):
    """Test that spyglass_chatvertexai wraps the LLM instance"""
    original_generate = mock_vertexai_llm._generate
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    assert wrapped_llm is mock_vertexai_llm
    assert wrapped_llm._generate is not original_generate


@patch("spyglass_ai.langchain_google_vertexai.spyglass_tracer")
def test_generate_traces_invocation(mock_tracer, mock_vertexai_llm, mock_generation):
    """Test that _generate creates a span"""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    mock_vertexai_llm._generate.return_value = mock_generation
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    mock_message = Mock(spec=['content', 'tool_calls', '__class__'])
    mock_message.__class__.__name__ = "HumanMessage"
    mock_message.content = "test"
    mock_message.tool_calls = []
    messages = [mock_message]
    wrapped_llm._generate(messages)
    
    mock_tracer.start_as_current_span.assert_called_once_with(
        "vertexai.chat.generate", record_exception=False
    )
    mock_span.set_status.assert_called_once()


@patch("spyglass_ai.langchain_google_vertexai.spyglass_tracer")
def test_generate_sets_attributes(mock_tracer, mock_vertexai_llm, mock_generation):
    """Test that generate sets correct span attributes"""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    mock_vertexai_llm._generate.return_value = mock_generation
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    mock_message = Mock(spec=['content', 'tool_calls', '__class__'])
    mock_message.__class__.__name__ = "HumanMessage"
    mock_message.content = "test"
    mock_message.tool_calls = []
    messages = [mock_message]
    wrapped_llm._generate(messages)
    
    # Verify key attributes were set
    set_attribute_calls = {
        call[0][0]: call[0][1] 
        for call in mock_span.set_attribute.call_args_list
    }
    
    assert set_attribute_calls["gen_ai.operation.name"] == "chat"
    assert set_attribute_calls["gen_ai.system"] == "vertex_ai"
    assert set_attribute_calls["gen_ai.request.model"] == "gemini-2.0-flash-exp"
    assert set_attribute_calls["gen_ai.request.vertex_ai.project"] == "test-project"
    assert set_attribute_calls["gen_ai.request.vertex_ai.location"] == "us-central1"
    assert set_attribute_calls["gen_ai.request.temperature"] == 0.7
    assert set_attribute_calls["gen_ai.request.max_tokens"] == 1024


@patch("spyglass_ai.langchain_google_vertexai.spyglass_tracer")
def test_generate_captures_token_usage(mock_tracer, mock_vertexai_llm, mock_generation):
    """Test that token usage is captured"""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    mock_vertexai_llm._generate.return_value = mock_generation
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    mock_message = Mock(spec=['content', 'tool_calls', '__class__'])
    mock_message.__class__.__name__ = "HumanMessage"
    mock_message.content = "test"
    mock_message.tool_calls = []
    messages = [mock_message]
    wrapped_llm._generate(messages)
    
    set_attribute_calls = {
        call[0][0]: call[0][1] 
        for call in mock_span.set_attribute.call_args_list
    }
    
    assert set_attribute_calls["gen_ai.usage.input_tokens"] == 10
    assert set_attribute_calls["gen_ai.usage.output_tokens"] == 20
    assert set_attribute_calls["gen_ai.usage.total_tokens"] == 30


@patch("spyglass_ai.langchain_google_vertexai.spyglass_tracer")
def test_generate_handles_exceptions(mock_tracer, mock_vertexai_llm):
    """Test that exceptions are properly recorded"""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    test_error = Exception("Test error")
    mock_vertexai_llm._generate = Mock(side_effect=test_error)
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    mock_message = Mock(spec=['content', 'tool_calls', '__class__'])
    mock_message.__class__.__name__ = "HumanMessage"
    mock_message.content = "test"
    mock_message.tool_calls = []
    messages = [mock_message]
    
    with pytest.raises(Exception) as exc_info:
        wrapped_llm._generate(messages)
    
    assert exc_info.value is test_error
    mock_span.record_exception.assert_called_once_with(test_error)


def test_format_langchain_messages_human_message():
    """Test formatting human/user messages"""
    message = Mock()
    message.__class__.__name__ = "HumanMessage"
    message.content = "Hello"
    message.tool_calls = []
    
    formatted = _format_langchain_messages([message])
    
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "Hello"


def test_format_langchain_messages_ai_message():
    """Test formatting AI/assistant messages"""
    message = Mock()
    message.__class__.__name__ = "AIMessage"
    message.content = "Hi there"
    message.tool_calls = []
    
    formatted = _format_langchain_messages([message])
    
    assert len(formatted) == 1
    assert formatted[0]["role"] == "assistant"
    assert formatted[0]["content"] == "Hi there"


def test_format_langchain_messages_with_tool_calls():
    """Test formatting messages with tool calls"""
    message = Mock()
    message.__class__.__name__ = "AIMessage"
    message.content = ""
    message.tool_calls = [
        {
            "id": "call_123",
            "name": "search",
            "args": {"query": "test"}
        }
    ]
    
    formatted = _format_langchain_messages([message])
    
    assert len(formatted) == 1
    assert "tool_calls" in formatted[0]
    assert len(formatted[0]["tool_calls"]) == 1
    assert formatted[0]["tool_calls"][0]["id"] == "call_123"
    assert formatted[0]["tool_calls"][0]["function"]["name"] == "search"


def test_format_langchain_messages_complex_content():
    """Test formatting messages with complex content"""
    message = Mock()
    message.__class__.__name__ = "HumanMessage"
    message.content = [
        {"type": "text", "text": "Part 1"},
        {"type": "text", "text": "Part 2"}
    ]
    message.tool_calls = []
    
    formatted = _format_langchain_messages([message])
    
    assert len(formatted) == 1
    assert "Part 1" in formatted[0]["content"]
    assert "Part 2" in formatted[0]["content"]


@pytest.mark.asyncio
@patch("spyglass_ai.langchain_google_vertexai.spyglass_tracer")
async def test_agenerate_traces_async_invocation(mock_tracer, mock_vertexai_llm, mock_generation):
    """Test that _agenerate creates a span for async calls"""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    # Mock async _agenerate
    async def mock_async_generate(*args, **kwargs):
        return mock_generation
    
    mock_vertexai_llm._agenerate = mock_async_generate
    
    wrapped_llm = spyglass_chatvertexai(mock_vertexai_llm)
    
    mock_message = Mock(spec=['content', 'tool_calls', '__class__'])
    mock_message.__class__.__name__ = "HumanMessage"
    mock_message.content = "test"
    mock_message.tool_calls = []
    messages = [mock_message]
    await wrapped_llm._agenerate(messages)
    
    mock_tracer.start_as_current_span.assert_called_with(
        "vertexai.chat.agenerate", record_exception=False
    )

