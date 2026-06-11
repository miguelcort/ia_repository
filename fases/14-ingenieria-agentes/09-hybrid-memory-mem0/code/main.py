"""
Lección: 09-hybrid-memory-mem0
Fase: 14
Mem0 (2024): hybrid memory layer para AI. Additive memory
extraccion + retrieval. Long-term + short-term. Vector + graph.
+Personalization, +scalability.
"""
from __future__ import annotations
import time
import math


def cosine_sim(a, b):
    """Cosine sim between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Mem0Memory:
    """Mock Mem0 hybrid memory."""
    def __init__(self, embed_dim=128):
        self.memories = []  # list of {id, content, embedding, timestamp}
        self.embed_dim = embed_dim
        self.short_term = []  # recent
        self.short_term_max = 5

    def _embed(self, content):
        """Mock embedding (deterministic)."""
        rng = sum(ord(c) for c in str(content))
        emb = []
        for i in range(self.embed_dim):
            v = ((rng + i * 17) % 100) / 100.0
            emb.append(v)
        return emb

    def add(self, content, metadata=None):
        """Add memory with embedding."""
        memory = {
            "id": len(self.memories),
            "content": content,
            "embedding": self._embed(content),
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.memories.append(memory)
        # add to short term
        self.short_term.append(memory)
        if len(self.short_term) > self.short_term_max:
            self.short_term.pop(0)
        return memory["id"]

    def search(self, query, n=3):
        """Search memories by query (cosine sim)."""
        q_emb = self._embed(query)
        scored = []
        for m in self.memories:
            score = cosine_sim(q_emb, m["embedding"])
            scored.append((m, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored[:n]]

    def get_short_term(self):
        return self.short_term

    def get_all(self):
        return self.memories


def add_to_graph(memory, entities, relations):
    """Mock: add entities and relations to graph."""
    return {
        "entities": entities,
        "relations": relations,
        "memory_id": memory,
    }


def main() -> int:
    mem = Mem0Memory(embed_dim=32)
    mem.add("User likes Python")
    mem.add("User works on AI")
    mem.add("User prefers dark mode")
    results = mem.search("Python", n=2)
    print(f"Top 2: {[m['content'] for m in results]}")
    print(f"Short term: {len(mem.get_short_term())} items")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())