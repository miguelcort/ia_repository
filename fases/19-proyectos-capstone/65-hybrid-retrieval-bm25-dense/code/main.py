"""
Lección: 65-hybrid-retrieval-bm25-dense
Fase: 19
Capstone de ingeniería AI: 65 Hybrid Retrieval Bm25 Dense.
"""
from __future__ import annotations
import sys

class HybridRetriever:
    def __init__(self, docs, alpha=0.5):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([d.split() for d in docs])
        self.docs = docs
        self.alpha = alpha

    def query(self, q, top_k=10):
        bm25_scores = self.bm25.get_scores(q.split())
        return list(zip(self.docs, bm25_scores))[:top_k]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
