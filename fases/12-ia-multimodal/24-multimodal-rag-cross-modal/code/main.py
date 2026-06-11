"""
Lección: 24-multimodal-rag-cross-modal
Fase: 12
Multimodal RAG cross-modal: text + image + audio + video retrieval.
Multimodal embeddings, hybrid search, re-ranking, generation.
GPT-4V RAG, VLM RAG pipelines.
"""
from __future__ import annotations
import numpy as np


def encode_text(text, embed_dim=512):
    """Text -> embedding."""
    rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
    return rng.standard_normal(embed_dim) * 0.1


def encode_image(image, embed_dim=512):
    """Image -> embedding (ViT/CLIP-style)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    return rng.standard_normal(embed_dim) * 0.1


def encode_audio(audio, embed_dim=512):
    """Audio -> embedding."""
    rng = np.random.default_rng(hash(audio.tobytes()[:64]) & 0xFFFFFFFF)
    return rng.standard_normal(embed_dim) * 0.1


def encode_video_frame(frame, embed_dim=512):
    """Video frame -> embedding."""
    return encode_image(frame, embed_dim=embed_dim)


def cosine_sim(a, b):
    """Cosine sim."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def multimodal_index(docs):
    """Index documents (mix of text/image/audio/video)."""
    embeddings = []
    metas = []
    for doc in docs:
        if doc["type"] == "text":
            emb = encode_text(doc["content"])
        elif doc["type"] == "image":
            emb = encode_image(doc["content"])
        elif doc["type"] == "audio":
            emb = encode_audio(doc["content"])
        elif doc["type"] == "video_frame":
            emb = encode_video_frame(doc["content"])
        embeddings.append(emb)
        metas.append(doc)
    return np.stack(embeddings), metas


def multimodal_search(index_emb, index_meta, query, top_k=3):
    """Search multimodal index."""
    if query["type"] == "text":
        q_emb = encode_text(query["content"])
    elif query["type"] == "image":
        q_emb = encode_image(query["content"])
    else:
        raise ValueError(f"Unsupported query type: {query['type']}")
    sims = index_emb @ q_emb / (np.linalg.norm(index_emb, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    top_idx = np.argsort(-sims)[:top_k]
    return [(index_meta[i], float(sims[i])) for i in top_idx]


def hybrid_search(index_emb, index_meta, query, keyword_index, top_k=3,
                  alpha=0.5):
    """Hybrid: vector + keyword (BM25-style)."""
    vector_results = multimodal_search(index_emb, index_meta, query, top_k=top_k * 2)
    # keyword: count of shared words
    q_words = set(query["content"].lower().split())
    keyword_scores = []
    for i, meta in enumerate(index_meta):
        if meta["type"] != "text":
            keyword_scores.append(0.0)
        else:
            words = set(meta["content"].lower().split())
            keyword_scores.append(len(q_words & words))
    # normalize
    if max(keyword_scores) > 0:
        keyword_scores = [s / max(keyword_scores) for s in keyword_scores]
    # combine
    vector_scores = np.array([score for _, score in vector_results])
    # re-rank
    combined = []
    for meta, vec_score in vector_results:
        idx = index_meta.index(meta)
        kw_score = keyword_scores[idx]
        combined.append((meta, alpha * vec_score + (1 - alpha) * kw_score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


def rerank(results, query, reranker_fn):
    """Re-rank results con un reranker (mock)."""
    reranked = sorted(results, key=lambda x: reranker_fn(query, x[0]), reverse=True)
    return reranked


def main() -> int:
    rng = np.random.default_rng(0)
    docs = [
        {"type": "text", "content": "A cat is sitting on a chair"},
        {"type": "image", "content": rng.standard_normal((224, 224, 3))},
        {"type": "text", "content": "Dogs are friendly animals"},
    ]
    index_emb, index_meta = multimodal_index(docs)
    print(f"Indexed {len(docs)} docs, emb shape: {index_emb.shape}")
    query = {"type": "text", "content": "cat chair"}
    results = multimodal_search(index_emb, index_meta, query, top_k=2)
    print(f"Top 2 results: {[(r[0]['type'], r[1]) for r in results]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())