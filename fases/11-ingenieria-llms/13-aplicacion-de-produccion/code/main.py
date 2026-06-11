"""
Lección: 13-aplicacion-de-produccion
Fase: 11
Production LLM apps: FastAPI, async, streaming, error handling, observability.
Best practices, deployment, scaling.
"""
from __future__ import annotations
import sys
import numpy as np


def chunk_response(text, chunk_size=20):
    """Chunk text para streaming."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def retry_with_backoff(max_retries=3, base_delay=1):
    """Mock retry con exponential backoff."""
    delays = [base_delay * (2 ** i) for i in range(max_retries)]
    return delays


def timeout_seconds(request_type="simple", model_size="7B"):
    """Mock timeout based on request type y model size."""
    if model_size == "7B":
        return 30 if request_type == "simple" else 120
    elif model_size == "70B":
        return 120 if request_type == "simple" else 600
    return 60


def rate_limit_check(requests, max_per_minute=60):
    """Mock rate limit: count requests en ultimo minuto."""
    if len(requests) > max_per_minute:
        return False
    return True


def error_responses():
    """Common LLM API error responses."""
    return {
        "rate_limit": {"status": 429, "error": "Rate limit exceeded"},
        "timeout": {"status": 408, "error": "Request timeout"},
        "context_length": {"status": 400, "error": "Context length exceeded"},
        "auth": {"status": 401, "error": "Invalid API key"},
        "model": {"status": 404, "error": "Model not found"},
        "server": {"status": 500, "error": "Internal server error"},
        "overloaded": {"status": 503, "error": "Service overloaded"},
    }


def main() -> int:
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_response(text, chunk_size=10)
    print(f"Streaming chunks: {len(chunks)}")
    delays = retry_with_backoff(3)
    print(f"Retry delays: {delays}")
    return 0


if __name__ == "__main__":
    sys.exit(main())