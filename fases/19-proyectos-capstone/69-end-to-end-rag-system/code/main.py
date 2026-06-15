"""
Lección: 69-end-to-end-rag-system
Fase: 19
Capstone de ingeniería AI: 69 End To End Rag System.
"""
from __future__ import annotations
import sys

class ProductionRAG:
    def __init__(self, llm, retriever, reranker):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker

    def query(self, q, top_k=10):
        hyp = self.llm.hyde(q)
        candidates = self.retriever.query(hyp, top_k=50)
        ranked = self.reranker.rerank(q, candidates, top_k=5)
        return self.llm.synthesize(q, ranked)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
