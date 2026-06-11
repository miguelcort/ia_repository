"""
Lección: 23-colpali-vision-native-rag
Fase: 12
ColPali (Faysse 2024): vision-native RAG. Document pages como
imagenes directas al VLM. PaliGemma + ColBERT-style late
interaction. SmolVLM, ColQwen2.
"""
from __future__ import annotations
import numpy as np


def colbert_late_interaction(doc_tokens, query_tokens, embed_dim=128):
    """ColBERT-style: MaxSim entre query y doc tokens.
    doc_tokens: (n_doc, d), query_tokens: (n_query, d)."""
    n_query = query_tokens.shape[0]
    n_doc = doc_tokens.shape[0]
    # similarity matrix
    sim = query_tokens @ doc_tokens.T
    # max over doc for each query
    max_sim = sim.max(axis=1)
    return float(max_sim.sum())


def encode_page_pali_gemma(image, n_patches=256, embed_dim=1152):
    """PaliGemma encode page -> (n_patches, embed_dim)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    return rng.standard_normal((n_patches, embed_dim)) * 0.1


def encode_query(query, n_tokens=10, embed_dim=1152):
    """Encode query."""
    rng = np.random.default_rng(hash(query) & 0xFFFFFFFF)
    return rng.standard_normal((n_tokens, embed_dim)) * 0.1


def colpali_retrieve(pages, query, top_k=3):
    """Retrieve top-k pages para query. pages: list of images."""
    q_emb = encode_query(query)
    scores = []
    for i, page in enumerate(pages):
        p_emb = encode_page_pali_gemma(page)
        score = colbert_late_interaction(p_emb, q_emb)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def colpali_rag(pages, query, generation_fn):
    """RAG: retrieve top pages + generate answer."""
    retrieved = colpali_retrieve(pages, query, top_k=3)
    context = "\n".join([f"Page {idx}: ..." for idx, _ in retrieved])
    prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
    return generation_fn(prompt)


def colqwen2_smol_vlm(pages, query):
    """ColQwen2 + SmolVLM: vision-native RAG optimized."""
    return colpali_retrieve(pages, query, top_k=5)


def main() -> int:
    rng = np.random.default_rng(0)
    pages = [rng.standard_normal((512, 512, 3)) for _ in range(10)]
    retrieved = colpali_retrieve(pages, "What is the revenue?", top_k=3)
    print(f"Top 3 pages: {[idx for idx, _ in retrieved]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())