"""
Lección: 66-reranker-cross-encoder
Fase: 19
Capstone de ingeniería AI: 66 Reranker Cross Encoder.
"""
from __future__ import annotations
import sys

def cross_encoder_rerank(query, docs, top_k=5,
                       model="BAAI/bge-reranker-v2-m3"):
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(model)
    pairs = [[query, d] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    return ranked[:top_k]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
