# Spyglass SDK Examples

Example scripts showing Spyglass integration with LLM providers.

## Setup

```bash
# Install for Vertex AI
pip install spyglass-ai[langchain-google-vertexai]

# Set environment variables
export SPYGLASS_API_KEY=your-api-key
export SPYGLASS_DEPLOYMENT_ID=your-deployment-id
export VERTEX_PROJECT_ID=your-gcp-project
export VERTEX_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Examples

```bash
# Basic Gemini usage
python basic_gemini_example.py

# Gemini with MCP tools
python gemini_mcp_example.py

# RAG with query classification
python gemini_rag_example.py
```

