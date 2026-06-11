"""
Lección: 17-tradeoffs-de-frameworks-de-agentes
Fase: 11
Comparacion de agentic frameworks: LangGraph, LlamaIndex, LangChain,
DSPy, Letta, Mastra, OpenAI Agents.
"""
from __future__ import annotations
import sys
import numpy as np


def framework_comparison():
    """Comparacion de agentic frameworks."""
    return {
        "LangGraph": {
            "type": "Stateful workflows",
            "language": "Python",
            "memory": "PostgresSaver, Redis, custom",
            "multi-actor": True,
            "production": True,
            "license": "MIT",
        },
        "LangChain": {
            "type": "LCEL chains, agents",
            "language": "Python, JS",
            "memory": "Built-in, custom",
            "multi-actor": True,
            "production": True,
            "license": "MIT",
        },
        "LlamaIndex": {
            "type": "RAG, agents",
            "language": "Python",
            "memory": "ChatMemoryBuffer, custom",
            "multi-actor": True,
            "production": True,
            "license": "MIT",
        },
        "DSPy": {
            "type": "Programmatic prompts",
            "language": "Python",
            "memory": "Custom",
            "multi-actor": False,
            "production": True,
            "license": "Apache 2.0",
        },
        "Letta": {
            "type": "Memory + agents",
            "language": "Python",
            "memory": "Built-in (mem0-style)",
            "multi-actor": True,
            "production": True,
            "license": "Apache 2.0",
        },
        "Mastra": {
            "type": "TypeScript agents",
            "language": "TypeScript",
            "memory": "Built-in",
            "multi-actor": True,
            "production": True,
            "license": "Apache 2.0",
        },
        "OpenAI Agents SDK": {
            "type": "OpenAI-specific agents",
            "language": "Python",
            "memory": "Sessions, custom",
            "multi-actor": True,
            "production": True,
            "license": "Apache 2.0",
        },
    }


def decision_matrix(requirements):
    """Decision matrix: 'stateful' -> LangGraph, etc."""
    matrix = {
        "stateful": "LangGraph",
        "memory": "Letta, mem0",
        "rag": "LlamaIndex",
        "typescript": "Mastra",
        "prompt-optimization": "DSPy",
        "openai": "OpenAI Agents SDK",
        "general": "LangChain",
    }
    return matrix.get(requirements, "LangChain")


def main() -> int:
    f = framework_comparison()
    for name, props in f.items():
        print(f"{name}: {props['type']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())