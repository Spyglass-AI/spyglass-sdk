# Spyglass Examples

This directory contains examples demonstrating how to use Spyglass for AI observability.

## Simple Bedrock + MCP Example

The `langchain_aws_mcp_example.py` demonstrates:
- ChatBedrockConverse integration with Spyglass tracing
- MCP (Model Context Protocol) tools integration  
- OpenTelemetry GenAI semantic conventions
- Async LLM calls with tool usage

### Quick Start

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env with your credentials:**
   ```bash
   # Required - get from https://app.spyglass-ai.com/
   SPYGLASS_DEPLOYMENT_ID=your-deployment-id-here
   SPYGLASS_API_KEY=your-spyglass-api-key-here
   
   # Required - AWS credentials for Bedrock
   AWS_ACCESS_KEY_ID=your-aws-access-key-id
   AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
   AWS_REGION=us-west-2
   ```

3. **Install dependencies:**
   ```bash
   pip install python-dotenv langchain-aws langchain-mcp-adapters
   ```

4. **Ensure Node.js is available:**
   ```bash
   node --version  # Should show v16+ 
   npm --version   # Should show npm version
   ```

5. **Run the example:**
   ```bash
   python langchain_aws_mcp_example.py
   ```

### What Gets Traced

The example will create traces with:
- `gen_ai.operation.name`: "chat"
- `gen_ai.system`: "aws_bedrock" 
- `gen_ai.request.model`: Model ID
- `gen_ai.input.messages`: Full input conversation
- `gen_ai.output.messages`: Full LLM response
- `gen_ai.usage.*`: Token usage statistics
- `gen_ai.request.tools.*`: Tool information
- `gen_ai.response.tools.*`: Tool call results

### Troubleshooting

- **"No MCP tools available"**: Ensure Node.js is installed and /tmp directory exists
- **AWS credentials error**: Check your AWS keys and Bedrock access permissions  
- **Spyglass connection error**: Verify your API key and deployment ID
- **Import errors**: Install missing dependencies with pip

### View Traces

After running the example, view traces at: https://app.spyglass-ai.com/
