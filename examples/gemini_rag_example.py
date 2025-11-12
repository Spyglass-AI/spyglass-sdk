#!/usr/bin/env python3
"""RAG example with Gemini and Spyglass."""

import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from spyglass_ai import spyglass_chatvertexai, spyglass_trace

load_dotenv()


class RAGService:
    def __init__(self):
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=os.getenv("VERTEX_PROJECT_ID", "dojo-iam-managed"),
            location=os.getenv("VERTEX_LOCATION", "us-central1")
        )
        
        self.llm = ChatVertexAI(
            model_name="gemini-2.0-flash-exp",
            project=os.getenv("VERTEX_PROJECT_ID", "dojo-iam-managed"),
            location=os.getenv("VERTEX_LOCATION", "us-central1"),
            temperature=0.7,
            max_output_tokens=1024,
        )
        
        # Wrap LLM with Spyglass tracing
        self.traced_llm = spyglass_chatvertexai(self.llm)
    
    @spyglass_trace(name="classify_query_intent")
    def classify_query_intent(self, query: str) -> dict:
        query_lower = query.lower()
        
        structured_keywords = [
            "latency", "error rate", "token usage", "cost", "requests",
            "throughput", "p95", "p99", "uptime", "failures"
        ]
        
        document_keywords = [
            "incident", "root cause", "why", "what caused",
            "explain", "how to", "configuration", "deployment"
        ]
        
        structured_score = sum(1 for kw in structured_keywords if kw in query_lower)
        document_score = sum(1 for kw in document_keywords if kw in query_lower)
        
        if structured_score > document_score:
            intent = "STRUCTURED_DATA"
            confidence = min(0.9, 0.6 + (structured_score * 0.1))
        elif document_score > structured_score:
            intent = "DOCUMENT_SEARCH"
            confidence = min(0.9, 0.6 + (document_score * 0.1))
        else:
            intent = "MIXED"
            confidence = 0.6
        
        return {"intent": intent, "confidence": confidence}
    
    @spyglass_trace(name="search_documents")
    def search_documents(self, query: str, limit: int = 5) -> list:
        # In real implementation, this would search a vector database
        return [
            {
                "content": "Summarization endpoint saw 3x increase in average tokens per request after prompt template change",
                "source": "incident_reports.md",
                "similarity": 0.92
            },
            {
                "content": "Token usage patterns show users are including full document context instead of chunks",
                "source": "usage_analysis.md",
                "similarity": 0.85
            }
        ]
    
    @spyglass_trace(name="query_llm")
    def query_llm(self, query: str, context: str) -> str:
        prompt = f"""
Based on the following context, answer this question about our AI system:

Question: {query}

Context:
{context}

Provide a clear technical answer.
"""
        
        messages = [
            SystemMessage(content="You are an AI observability expert analyzing system metrics."),
            HumanMessage(content=prompt)
        ]
        
        response = self.traced_llm.invoke(messages)
        return response.content
    
    @spyglass_trace(name="process_query")
    def process_query(self, query: str) -> dict:
        # Step 1: Classify intent
        classification = self.classify_query_intent(query)
        print(f"Query classified as: {classification['intent']}")
        
        # Step 2: Search documents
        documents = self.search_documents(query)
        
        # Step 3: Build context
        context = "\n\n".join([doc["content"] for doc in documents])
        
        # Step 4: Query LLM
        answer = self.query_llm(query, context)
        
        return {
            "answer": answer,
            "sources": documents,
            "query_type": classification["intent"]
        }


def main():
    rag = RAGService()
    
    query = "What's causing the spike in token usage for our summarization endpoint?"
    print(f"Query: {query}\n")
    
    result = rag.process_query(query)
    
    print(f"Answer: {result['answer']}\n")
    print(f"Query Type: {result['query_type']}")
    print(f"Sources: {len(result['sources'])} documents")


if __name__ == "__main__":
    main()

