"""
Example: Simple LangChain AWS Bedrock + MCP Tool Integration with Spyglass Tracing

This example demonstrates a simple use case of Spyglass tracing for ChatBedrockConverse
with a single async call using MCP tools.

Setup:
1. Copy .env.example to .env: cp .env.example .env
2. Edit .env with your Spyglass and AWS credentials
3. Ensure Node.js is installed for the MCP filesystem server
4. Run: python langchain_aws_mcp_example.py

Note: With the updated Spyglass SDK, imports can be done in any order - environment
variables are loaded lazily when needed.
"""

import asyncio
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "Warning: python-dotenv not installed. Environment variables must be set manually."
    )
    print("Install with: pip install python-dotenv")

# Spyglass imports - can now be imported before loading environment variables
from spyglass_ai import spyglass_chatbedrockconverse, spyglass_mcp_tools_async

# LangChain imports
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

# MCP imports
from langchain_mcp_adapters.sessions import create_session


async def simple_bedrock_mcp_example():
    """Simple example with ChatBedrockConverse, ainvoke, and MCP tools."""

    # 1. Create ChatBedrockConverse instance
    llm = ChatBedrockConverse(
        model="qwen.qwen3-32b-v1:0",
        region_name="us-west-2",
        temperature=0.1,
        max_tokens=1000,
    )

    # 2. Wrap with Spyglass tracing
    traced_llm = spyglass_chatbedrockconverse(llm)

    # 3. Setup MCP connection (using a simple example MCP server)
    # Note: This example assumes you have an MCP server running
    # For demo purposes, we'll use a filesystem MCP server
    connection = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    }

    try:
        # 4. Create MCP session and load tools
        async with create_session(connection) as session:
            try:
                await session.initialize()
            except Exception as init_error:
                print(f"❌ Error during session initialization: {init_error}")
                print(f"Error type: {type(init_error).__name__}")
                import traceback

                traceback.print_exc()
                return

            try:
                # 5. Load and trace MCP tools
                traced_tools = await spyglass_mcp_tools_async(session=session)
            except Exception as tools_error:
                print(f"❌ Error loading MCP tools: {tools_error}")
                print(f"Error type: {type(tools_error).__name__}")
                import traceback

                traceback.print_exc()
                return

            if not traced_tools:
                print("❌ No MCP tools available from server")
                return

            # 6. Bind MCP tools to the LLM
            llm_with_tools = traced_llm.bind_tools(traced_tools)

            # 7. Create messages with a task that can use available tools
            messages = [
                SystemMessage(
                    content="You are a helpful assistant with access to filesystem tools."
                ),
                HumanMessage(
                    content="Can you help me understand what tools are available? List the available tools and their purposes."
                ),
            ]

            # 8. Make a single async call (this will be traced by Spyglass)
            try:
                response = await llm_with_tools.ainvoke(messages)
                print(f"Response: {response.content}")
            except Exception as llm_error:
                print(f"❌ Error during LLM call: {llm_error}")
                print(f"Error type: {type(llm_error).__name__}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        print("\nNote: This requires:")
        print("  1. Valid AWS credentials and Bedrock access")
        print("  2. Node.js and npm installed for MCP server")
        print("  3. Internet connection to download MCP server")
        print("  4. /tmp directory accessible")
        print("\nTo install Node.js MCP server manually:")
        print("  npm install -g @modelcontextprotocol/server-filesystem")


async def main():
    """Run the simple example"""
    # Check if required environment variables are set
    if not os.getenv("SPYGLASS_DEPLOYMENT_ID"):
        print("❌ SPYGLASS_DEPLOYMENT_ID environment variable is required")
        print("\nQuick setup:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Edit .env and add your Spyglass deployment ID")
        print("3. Get your deployment ID from: https://app.spyglass-ai.com/")
        return

    if not os.getenv("SPYGLASS_API_KEY"):
        print("❌ SPYGLASS_API_KEY environment variable is required")
        print("\nQuick setup:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Edit .env and add your Spyglass API key")
        print("3. Get your API key from: https://app.spyglass-ai.com/")
        return

    # Check AWS credentials
    if not (os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE")):
        print("⚠️  Warning: No AWS credentials found")
        print("Please set either:")
        print("- AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env, or")
        print("- AWS_PROFILE in .env to use a configured AWS profile")
        print("\nContinuing anyway - this may fail when calling Bedrock...")

    # Run the simple example
    await simple_bedrock_mcp_example()


if __name__ == "__main__":
    asyncio.run(main())
