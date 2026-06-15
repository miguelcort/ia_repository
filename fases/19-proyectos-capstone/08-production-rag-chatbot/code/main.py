"""
Lección: 08-production-rag-chatbot
Fase: 19
Capstone de ingeniería AI: 08 Production Rag Chatbot.
"""
from __future__ import annotations
import sys

class ProductionRAG:
    def __init__(self, llm, embedder, reranker):
        self.llm = llm
        self.embedder = embedder
        self.reranker = reranker

    def query(self, question, top_k=10):
        dense = self.dense_retrieve(question, top_k)
        bm25 = self.bm25_retrieve(question, top_k)
        candidates = self.merge_unique(dense, bm25)
        ranked = self.reranker.rerank(question, candidates, top_k=5)
        return self.llm.synthesize(question, ranked)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
