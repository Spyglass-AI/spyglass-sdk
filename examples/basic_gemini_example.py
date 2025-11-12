#!/usr/bin/env python3
"""Basic example using Spyglass with Gemini."""

import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from spyglass_ai import spyglass_chatvertexai, spyglass_trace

load_dotenv()


@spyglass_trace(name="classify_query")
def classify_query(query: str) -> str:
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ["latency", "error", "token", "cost", "p95", "p99"]):
        return "METRICS"
    elif any(keyword in query_lower for keyword in ["incident", "root cause", "why", "explain"]):
        return "DIAGNOSTIC"
    else:
        return "GENERAL"


@spyglass_trace(name="analyze_with_gemini")
def analyze_with_gemini(query: str, context: str) -> str:
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash-exp",
        project=os.getenv("VERTEX_PROJECT_ID", "dojo-iam-managed"),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
        temperature=0.7,
        max_output_tokens=1024,
    )
    
    # Wrap with Spyglass tracing
    traced_llm = spyglass_chatvertexai(llm)
    
    messages = [
        SystemMessage(content="You are an AI observability expert analyzing system performance."),
        HumanMessage(
            content=f"Based on this context:\n\n{context}\n\nAnswer: {query}"
        ),
    ]
    
    response = traced_llm.invoke(messages)
    return response.content


def main():
    query = "What are our P95 latency trends and which endpoints are slowest?"
    
    # Classify query
    query_type = classify_query(query)
    print(f"Query type: {query_type}")
    
    # Simulate some context data
    context = """
    Last 30 days API Performance:
    - P50 latency: 145ms
    - P95 latency: 890ms
    - P99 latency: 2.1s
    - Slowest endpoint: /api/search (1.8s P95)
    - Error rate: 0.8%
    """
    
    # Analyze with Gemini
    answer = analyze_with_gemini(query, context)
    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()

