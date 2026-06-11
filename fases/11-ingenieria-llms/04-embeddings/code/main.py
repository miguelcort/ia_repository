"""
Lección: 04-embeddings
Fase: 11
Embeddings: word, sentence, document embeddings. Cosine similarity.
OpenAI, Sentence-Transformers, Cohere, BGE, E5, MTEB.
"""
from __future__ import annotations
import sys
import numpy as np


def normalize(v):
    """L2 normalize."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-9)


def cosine_similarity(a, b):
    """Cosine sim entre dos vectores."""
    a_n = normalize(a.reshape(1, -1))[0]
    b_n = normalize(b.reshape(1, -1))[0]
    return float(np.dot(a_n, b_n))


def dot_product(a, b):
    return float(np.dot(a, b))


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def embed_simple(text, dim=64):
    """Mock embedding: hash-based projection."""
    rng = np.random.default_rng(hash(text) % (2**32))
    return rng.standard_normal(dim) * 0.1


def top_k_similar(query, corpus, k=3):
    """Top-k most similar to query en corpus."""
    sims = [(i, cosine_similarity(query, c)) for i, c in enumerate(corpus)]
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


def main() -> int:
    q = embed_simple("hola")
    corpus = [embed_simple(t) for t in ["hola", "adios", "hello"]]
    top = top_k_similar(q, corpus, k=2)
    print(f"Top-2 similar: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())