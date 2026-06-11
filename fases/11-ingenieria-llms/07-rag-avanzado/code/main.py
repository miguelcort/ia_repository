"""
Lección: 07-rag-avanzado
Fase: 11
Advanced RAG: query rewriting, multi-query, HyDE, re-ranking, hybrid.
Self-RAG, CRAG, GraphRAG.
"""
from __future__ import annotations
import sys
import numpy as np


def hyde_generate_hypothetical(query):
    """Hypothetical Document Embeddings: generate hypothetical answer."""
    return f"Hipotetica respuesta a '{query}': " + "informacion relevante. " * 5


def multi_query_rewrite(query, n=3):
    """Multi-query: rephrase query en n variaciones."""
    return [
        query,
        f"Cual es la explicacion de {query}?",
        f"Describe {query} en detalle",
    ][:n]


def reciprocal_rank_fusion(rankings, k=60):
    """RRF: combinar multiples rankings. rankings: list of list of (doc_id, score)."""
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def cross_encoder_rerank(query, docs, mock_scores=None):
    """Cross-encoder: rerank docs por relevance.
    Mock: returns docs + scores.
    """
    if mock_scores is None:
        mock_scores = [0.9 - 0.1 * i for i in range(len(docs))]
    paired = list(zip(docs, mock_scores))
    paired.sort(key=lambda x: -x[1])
    return paired


def crag_score_relevance(query, doc):
    """Mock: relevance score 0-1."""
    return 0.8 if query.split()[0] in doc else 0.3


def main() -> int:
    print("=== Multi-query ===")
    queries = multi_query_rewrite("que es AI", n=3)
    print(queries)
    print("\n=== RRF ===")
    rankings = [
        [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
        [("doc2", 0.85), ("doc1", 0.75), ("doc4", 0.65)],
    ]
    fused = reciprocal_rank_fusion(rankings)
    print(fused)
    return 0


if __name__ == "__main__":
    sys.exit(main())