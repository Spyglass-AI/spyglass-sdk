# Spyglass SDK
The Spyglass SDK provides client code for shipping telemetry data to the Spyglass AI platform

## Usage

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