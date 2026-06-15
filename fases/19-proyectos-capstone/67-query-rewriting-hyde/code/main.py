"""
Lección: 67-query-rewriting-hyde
Fase: 19
Capstone de ingeniería AI: 67 Query Rewriting Hyde.
"""
from __future__ import annotations
import sys

def hyde_retrieval(query, llm, retriever, k=10):
    hyp = llm(f"Answer concisely: {query}")
    return retriever.query(hyp, top_k=k)


def multi_query(query, llm, retriever, n=4):
    variants = llm(f"Generate {n} alternative phrasings: {query}")
    return [retriever.query(v) for v in variants]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
