"""
Lección: 06-rag
Fase: 11
RAG (Retrieval-Augmented Generation): retrieve docs, augment prompt, generate.
Vector store, embeddings, LLM.
"""
from __future__ import annotations
import sys
import numpy as np


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_n, b_n))


def simple_vector_store(documents, embeddings):
    """Vector store: dict de doc_id -> embedding."""
    return dict(zip(documents, embeddings))


def retrieve(query_emb, store, top_k=3):
    """Retrieve top-k most similar documents."""
    sims = [(doc, cosine_sim(query_emb, emb)) for doc, emb in store.items()]
    sims.sort(key=lambda x: -x[1])
    return [doc for doc, _ in sims[:top_k]]


def build_rag_prompt(query, retrieved_docs):
    """Build RAG prompt."""
    context = "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(retrieved_docs))
    return f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta basada en el contexto:"


def rag_pipeline(query, store, embed_fn, top_k=3):
    """Full RAG pipeline: query -> embed -> retrieve -> augment -> LLM (mock)."""
    q_emb = embed_fn(query)
    docs = retrieve(q_emb, store, top_k=top_k)
    prompt = build_rag_prompt(query, docs)
    # Mock LLM: devolver ultimo doc
    return docs[-1] if docs else "Sin respuesta"


def main() -> int:
    docs = ["AI es la simulacion de inteligencia humana.", "ML es subset de AI.", "Python es un lenguaje."]
    embeds = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([0.0, 1.0])]
    store = simple_vector_store(docs, embeds)
    q = np.array([0.95, 0.05])
    top = retrieve(q, store, top_k=2)
    print(f"Top-2: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())