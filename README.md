# Spyglass SDK
The Spyglass SDK provides client code for shipping telemetry data to the Spyglass AI platform

## Installation
```bash
pip install spyglass-ai
```

## Configuration

Set the following environment variables to configure the SDK:

### Required
- `SPYGLASS_API_KEY`: Your Spyglass API key
- `SPYGLASS_DEPLOYMENT_ID`: Unique identifier for your deployment
  - **Note**: Used for both `service.name` and `deployment.id` attributes

### Optional
- `SPYGLASS_OTEL_EXPORTER_OTLP_ENDPOINT`: Custom endpoint for development

### Example Configuration
```bash
export SPYGLASS_API_KEY="your-api-key"
export SPYGLASS_DEPLOYMENT_ID="user-service-v1.2.0"  # Required - used for both service.name and deployment.id
```

**Note**: `SPYGLASS_DEPLOYMENT_ID` is required and will be used for both the OpenTelemetry `service.name` and `deployment.id` resource attributes. This ensures consistency and simplifies dashboard queries.

## Usage

### Basic Function Tracing

Use the `@spyglass_trace` decorator to automatically trace function calls:

```python
from spyglass_ai import spyglass_trace

@spyglass_trace()
def calculate_total(price, tax_rate):
    return price * (1 + tax_rate)

# Usage
result = calculate_total(100, 0.08)  # This call will be traced
```

You can also provide a custom span name:

```python
@spyglass_trace(name="payment_processing")
def process_payment(amount, card_info):
    # Payment processing logic
    return {"status": "success", "transaction_id": "tx_123"}
```

### OpenAI Integration

Wrap your OpenAI client to automatically trace API calls:

```python
from openai import OpenAI
from spyglass_ai import spyglass_openai

# Create your OpenAI client
client = OpenAI(api_key="your-api-key")

# Wrap it with Spyglass tracing
client = spyglass_openai(client)

# Now all chat completions will be traced
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

### Complete Example

```python
from openai import OpenAI
from spyglass_ai import spyglass_trace, spyglass_openai

# Set up OpenAI client with tracing
client = OpenAI(api_key="your-api-key")
client = spyglass_openai(client)

@spyglass_trace(name="ai_conversation")
def have_conversation(user_message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}],
        max_tokens=150
    )
    return response.choices[0].message.content

# This will create traces for both the function and the OpenAI API call
answer = have_conversation("What is the capital of France?")
print(answer)
```

## What Gets Traced

### Function Tracing (`@spyglass_trace`)
- Function name, module, and qualified name
- Input arguments (excluding `self` and `cls`)
- Return values
- Execution time
- Any exceptions that occur

### OpenAI Integration (`spyglass_openai`)
- Model used
- Number of messages
- Request parameters (max_tokens, temperature, etc.)
- Token usage (prompt, completion, total)
- Response model
- Any API errors

## Development
### Install Dependencies
```bash
uv sync --extra test
```

### Run All Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_trace.py -v
```