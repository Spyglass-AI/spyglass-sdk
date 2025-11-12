#!/usr/bin/env python3
"""Spyglass with Gemini and MCP tools."""

import os
import asyncio
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage
from spyglass_ai import spyglass_chatvertexai, spyglass_trace, spyglass_mcp_tools_async

load_dotenv()


@spyglass_trace(name="query_with_tools")
async def query_with_tools(query: str) -> str:
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-exp",
        project=os.getenv("VERTEX_PROJECT_ID", "dojo-iam-managed"),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
        temperature=0.3,
        max_output_tokens=2048,
    )
    
    # Wrap with Spyglass tracing
    traced_llm = spyglass_chatvertexai(llm)
    
    # Example MCP server URL (filesystem tools)
    mcp_server_url = "npx -y @modelcontextprotocol/server-filesystem /tmp"
    
    # Wrap MCP tools with Spyglass tracing
    tools = await spyglass_mcp_tools_async(mcp_server_url)
    
    # Bind tools to LLM
    llm_with_tools = traced_llm.bind_tools(tools)
    
    # Invoke with query
    response = llm_with_tools.invoke([HumanMessage(content=query)])
    
    return response.content


async def main():
    query = "Check the log files for any error patterns in the last hour"
    
    print(f"Query: {query}")
    answer = await query_with_tools(query)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())

