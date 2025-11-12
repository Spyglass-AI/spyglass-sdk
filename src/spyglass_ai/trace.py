import functools
import inspect
from typing import Any, Callable, Optional, Union

from opentelemetry.trace import Span, Status, StatusCode

from .otel import spyglass_tracer


def spyglass_trace(name: Optional[str] = None) -> Callable:
    """
    A decorator that adds tracing to a function.
    Always captures function arguments and return values as span attributes.

    Args:
        name: Custom span name. If None, uses the qualified function name.

    Returns:
        Decorator function that wraps the target method with tracing.

    Example:
        @spyglass_trace(name="my_operation")
        def my_method(self, param1, param2):
            return param1 + param2

        # Or with default naming
        @spyglass_trace()
        def another_method(x, y):
            return x * y
    """

    def decorator(func: Callable) -> Callable:
        # Check if function is async
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                """
                This function will be called in place of the original async function
                that has the @spyglass_trace decorator.
                """
                # Generate span name
                span_name = name or f"{func.__module__}.{func.__qualname__}"

                # Use global tracer, no need to create new instance each time
                # Set record_exception=False since we manually record exceptions in the except block
                with spyglass_tracer.start_as_current_span(
                    span_name, record_exception=False
                ) as span:
                    try:
                        # Set base attributes
                        _set_base_attributes(span, func)

                        # Capture arguments
                        _capture_arguments(span, func, args, kwargs)

                        # Execute the original async function
                        result = await func(*args, **kwargs)

                        # Capture return value
                        _capture_return_value(span, result)

                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        # Record exception and mark span as error
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            wrapper = async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                """
                This function will be called in place of the original function
                that has the @spyglass_trace decorator.
                """
                # Generate span name
                span_name = name or f"{func.__module__}.{func.__qualname__}"

                # Use global tracer, no need to create new instance each time
                # Set record_exception=False since we manually record exceptions in the except block
                with spyglass_tracer.start_as_current_span(
                    span_name, record_exception=False
                ) as span:
                    try:
                        # Set base attributes
                        _set_base_attributes(span, func)

                        # Capture arguments
                        _capture_arguments(span, func, args, kwargs)

                        # Execute the original function
                        result = func(*args, **kwargs)

                        # Capture return value
                        _capture_return_value(span, result)

                        # Mark span as successful
                        span.set_status(Status(StatusCode.OK))

                        return result

                    except Exception as e:
                        # Record exception and mark span as error
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            wrapper = sync_wrapper

        # Safely wrap function metadata, handling union type annotations
        # Try functools.update_wrapper first, but fall back to manual copying
        # if UnionType issues occur (Python 3.10+)
        try:
            functools.update_wrapper(wrapper, func)
        except (TypeError, AttributeError):
            # Fall back to manual attribute copying if update_wrapper fails
            # (e.g., due to UnionType issues in Python 3.10+)
            try:
                wrapper.__name__ = getattr(func, "__name__", "wrapper")
                wrapper.__doc__ = getattr(func, "__doc__", None)
                wrapper.__module__ = getattr(func, "__module__", None)
                wrapper.__qualname__ = getattr(func, "__qualname__", wrapper.__name__)
                # Preserve function signature for frameworks like FastAPI that inspect it
                if hasattr(func, "__signature__"):
                    wrapper.__signature__ = func.__signature__
                else:
                    # Try to get signature from inspect if __signature__ doesn't exist
                    try:
                        wrapper.__signature__ = inspect.signature(func)
                    except (ValueError, TypeError):
                        pass  # If signature can't be determined, continue without it
                # Preserve annotations for type checking and frameworks like FastAPI
                if hasattr(func, "__annotations__"):
                    wrapper.__annotations__ = func.__annotations__
            except (TypeError, AttributeError):
                pass  # If copying fails, continue without metadata

        return wrapper

    return decorator


def _set_base_attributes(span: Span, func: Callable) -> None:
    """Set base attributes on the span."""
    span.set_attribute("function.name", func.__name__)
    span.set_attribute("function.module", func.__module__)
    span.set_attribute("function.qualname", func.__qualname__)


def _capture_arguments(span: Span, func: Callable, args: tuple, kwargs: dict) -> None:
    """Capture function arguments as span attributes."""
    try:
        # Get function signature
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Set argument attributes
        for arg_name, arg_value in bound_args.arguments.items():
            # Skip 'self' and 'cls' parameters
            if arg_name in ("self", "cls"):
                continue

            attr_key = f"function.args.{arg_name}"
            span.set_attribute(attr_key, _serialize_attribute_value(arg_value))

    except Exception:
        # If argument capture fails, don't break the span
        span.set_attribute("function.args.capture_error", True)


def _capture_return_value(span: Span, return_value: Any) -> None:
    """Capture return value as span attribute."""
    try:
        span.set_attribute(
            "function.return_value", _serialize_attribute_value(return_value)
        )
    except Exception:
        # If return value capture fails, don't break the span
        span.set_attribute("function.return_value.capture_error", True)


def _serialize_attribute_value(value: Any) -> Union[str, int, float, bool]:
    """
    Serialize a value to be suitable for span attributes.
    OpenTelemetry attributes must be basic types.
    """
    if isinstance(value, str):
        return value[:1000]  # Limit string length to avoid huge attributes
    elif isinstance(value, (int, float, bool)):
        return value
    elif value is None:
        return "None"
    else:
        # Convert complex types to string representation
        try:
            return str(value)[:1000]  # Limit length to avoid huge attributes
        except Exception:
            return "<unable_to_serialize>"
