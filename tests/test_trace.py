import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.trace import StatusCode

from spyglass_ai.trace import (
    _capture_arguments,
    _capture_return_value,
    _serialize_attribute_value,
    _set_base_attributes,
    spyglass_trace,
)


class TestSpyglassTrace:
    """Test the spyglass_trace decorator."""

    @patch("spyglass_ai.trace.spyglass_tracer")
    def test_basic_function_tracing(self, mock_tracer):
        """Test basic function tracing with default name."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        @spyglass_trace()
        def test_function(x, y):
            return x + y

        result = test_function(1, 2)

        # Check that tracer was called with correct span name
        expected_name = f"{test_function.__module__}.{test_function.__qualname__}"
        mock_tracer.start_as_current_span.assert_called_once_with(
            expected_name, record_exception=False
        )

        # Check that result is correct
        assert result == 3

    @patch("spyglass_ai.trace.spyglass_tracer")
    def test_custom_span_name(self, mock_tracer):
        """Test custom span name."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        @spyglass_trace(name="test_function")
        def test_function():
            return "test"

        test_function()

        mock_tracer.start_as_current_span.assert_called_once_with(
            "test_function", record_exception=False
        )

    @patch("spyglass_ai.trace.spyglass_tracer")
    def test_exception_handling(self, mock_tracer):
        """Test that exceptions are properly recorded and re-raised."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        @spyglass_trace()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        # Check that exception was recorded
        mock_span.record_exception.assert_called_once()
        # Check status was set to ERROR with correct message
        call_args = mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.ERROR
        assert call_args.description == "Test error"

    @patch("spyglass_ai.trace.spyglass_tracer")
    def test_success_status(self, mock_tracer):
        """Test that successful execution sets OK status."""
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(
            return_value=None
        )

        @spyglass_trace()
        def success_function():
            return "success"

        success_function()

        # Check status was set to OK
        call_args = mock_span.set_status.call_args[0][0]
        assert call_args.status_code == StatusCode.OK


class TestHelperFunctions:
    """Test helper functions used by the decorator."""

    def test_set_base_attributes(self):
        """Test setting base function attributes."""
        mock_span = MagicMock()

        def test_func():
            pass

        _set_base_attributes(mock_span, test_func)

        expected_calls = [
            ("function.name", test_func.__name__),
            ("function.module", test_func.__module__),
            ("function.qualname", test_func.__qualname__),
        ]

        for expected_call in expected_calls:
            mock_span.set_attribute.assert_any_call(*expected_call)

    def test_capture_arguments_simple(self):
        """Test capturing simple function arguments."""
        mock_span = MagicMock()

        def test_func(a, b, c=None):
            pass

        _capture_arguments(mock_span, test_func, (1, 2), {"c": 3})

        expected_calls = [
            ("function.args.a", 1),
            ("function.args.b", 2),
            ("function.args.c", 3),
        ]

        for expected_call in expected_calls:
            mock_span.set_attribute.assert_any_call(*expected_call)

    def test_capture_arguments_with_self(self):
        """Test that 'self' and 'cls' parameters are skipped."""
        mock_span = MagicMock()

        def test_method(self, param):
            pass

        _capture_arguments(mock_span, test_method, ("self_obj", "value"), {})

        # Should only capture 'param', not 'self'
        mock_span.set_attribute.assert_called_once_with("function.args.param", "value")

    def test_capture_arguments_error_handling(self):
        """Test error handling in argument capture."""
        mock_span = MagicMock()

        # This should cause an error in signature binding
        def test_func(a):
            pass

        _capture_arguments(mock_span, test_func, (1, 2, 3), {})  # Too many args

        # Should set error flag instead of crashing
        mock_span.set_attribute.assert_called_with("function.args.capture_error", True)

    def test_capture_return_value(self):
        """Test capturing return values."""
        mock_span = MagicMock()

        _capture_return_value(mock_span, "test_result")

        mock_span.set_attribute.assert_called_once_with(
            "function.return_value", "test_result"
        )

    def test_capture_return_value_error_handling(self):
        """Test error handling in return value capture."""
        mock_span = MagicMock()

        # Mock set_attribute to raise an exception
        mock_span.set_attribute.side_effect = [Exception("Test error"), None]

        _capture_return_value(mock_span, "test_result")

        # Should handle the error and set error flag
        mock_span.set_attribute.assert_called_with(
            "function.return_value.capture_error", True
        )


class TestSerializeAttributeValue:
    """Test attribute value serialization."""

    def test_basic_types(self):
        """Test that basic types are returned as-is."""
        assert _serialize_attribute_value("string") == "string"
        assert _serialize_attribute_value(42) == 42
        assert _serialize_attribute_value(3.14) == 3.14
        assert _serialize_attribute_value(True) is True
        assert _serialize_attribute_value(False) is False

    def test_none_value(self):
        """Test None value serialization."""
        assert _serialize_attribute_value(None) == "None"

    def test_complex_types(self):
        """Test complex type serialization."""
        test_dict = {"key": "value"}
        result = _serialize_attribute_value(test_dict)
        assert isinstance(result, str)
        assert "key" in result and "value" in result

    def test_long_string_truncation(self):
        """Test that long strings are truncated."""
        long_string = "x" * 1500
        result = _serialize_attribute_value(long_string)
        assert len(result) == 1000

    def test_serialization_error_handling(self):
        """Test handling of objects that can't be serialized."""

        class UnserializableObject:
            def __str__(self):
                raise Exception("Can't serialize")

        obj = UnserializableObject()
        result = _serialize_attribute_value(obj)
        assert result == "<unable_to_serialize>"


class TestIntegration:
    """Integration tests with real OpenTelemetry components."""

    @patch.dict(
        os.environ,
        {
            "SPYGLASS_DEPLOYMENT_ID": "test-deployment",
            "SPYGLASS_API_KEY": "test-api-key",
        },
    )
    def test_real_tracing_integration(self):
        """Test the decorator with real OpenTelemetry setup."""

        @spyglass_trace(name="integration_test")
        def test_function(x, y=10):
            return x * y

        # This should work without mocking
        result = test_function(5, y=3)
        assert result == 15

    @patch.dict(
        os.environ,
        {
            "SPYGLASS_DEPLOYMENT_ID": "test-deployment",
            "SPYGLASS_API_KEY": "test-api-key",
        },
    )
    def test_class_method_tracing(self):
        """Test tracing class methods."""

        class TestClass:
            @spyglass_trace()
            def method(self, value):
                return value * 2

        obj = TestClass()
        result = obj.method(21)
        assert result == 42
