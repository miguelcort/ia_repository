"""
Lección: 14-prompt-semantic-caching
Fase: 17
Prompt semantic caching: cache LLM
responses by semantic similarity of
prompts, embeddings, threshold, hit rate,
fallback.
"""
from __future__ import annotations
import math


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.entries = []

    def add(self, embedding, response):
        self.entries.append({"embedding": embedding, "response": response})

    def lookup(self, query_embedding):
        best_sim = -1
        best_response = None
        for entry in self.entries:
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim and sim >= self.threshold:
                best_sim = sim
                best_response = entry["response"]
        if best_response is not None:
            return best_response, best_sim
        return None, 0.0

    def size(self):
        return len(self.entries)

    def hit_rate(self, total_queries, hits):
        if total_queries == 0:
            return 0.0
        return hits / total_queries


def main() -> int:
    cache = SemanticCache(threshold=0.9)
    cache.add([1.0, 0.0, 0.0], "response1")
    cache.add([0.0, 1.0, 0.0], "response2")
    response, sim = cache.lookup([0.95, 0.05, 0.0])
    print(f"Hit: {response}, sim: {sim:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())