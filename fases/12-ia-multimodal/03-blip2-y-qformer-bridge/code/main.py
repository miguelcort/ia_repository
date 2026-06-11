"""
Lección: 03-blip2-y-qformer-bridge
Fase: 12
BLIP-2 (Li 2023, Salesforce): frozen image encoder + frozen LLM + Q-Former bridge.
Q-Former: queries (32) cruzan attention con image features y text.
"""
from __future__ import annotations
import sys
import numpy as np


def qformer_queries(num_queries=32, query_dim=768, seed=0):
    """Inicializa queries (N, D) y pesos simples de cross-attention simulada."""
    rng = np.random.default_rng(seed)
    queries = rng.standard_normal((num_queries, query_dim)) * 0.02
    W_q = rng.standard_normal((query_dim, query_dim)) * 0.1
    W_kv = rng.standard_normal((query_dim, query_dim)) * 0.1
    return queries, W_q, W_kv


def cross_attention(queries, kv, W_q, W_kv):
    """Cross-attention: queries x kv. Returns (n_queries, query_dim)."""
    q = queries @ W_q
    k = kv @ W_kv[: kv.shape[1], :]
    # ajuste dim
    if k.shape[1] != q.shape[1]:
        W_kv2 = W_kv[:, : q.shape[1]]
        k = kv @ W_kv2
        v = kv @ W_kv2
    else:
        v = k
    d = q.shape[-1]
    scores = q @ k.T / np.sqrt(d)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)
    return weights @ v


def qformer_forward(image_features, queries, W_q, W_kv, n_layers=2):
    """Stack de cross-attention con image features (image + self-attn simplificado)."""
    h = queries
    for _ in range(n_layers):
        # cross-attn con image
        h = cross_attention(h, image_features, W_q, W_kv)
    return h


def blip2_forward(image_features, text_input_emb, queries, W_q, W_kv, llm_proj):
    """image -> Q-Former -> linear -> LLM space -> concat text -> LLM (simulado)."""
    z = qformer_forward(image_features, queries, W_q, W_kv)
    image_tokens = z @ llm_proj
    # concat: image tokens + text embeddings
    return np.vstack([image_tokens, text_input_emb])


def main() -> int:
    rng = np.random.default_rng(0)
    # image: 257 tokens (1 CLS + 256 patches) dim 768 (matched con Q-Former)
    image_features = rng.standard_normal((257, 768)) * 0.1
    queries, W_q, W_kv = qformer_queries(num_queries=32, query_dim=768, seed=0)
    z = qformer_forward(image_features, queries, W_q, W_kv, n_layers=2)
    print(f"Q-Former output: {z.shape} (32 queries, 768 dim)")
    # proyeccion a LLM dim 4096
    llm_proj = rng.standard_normal((768, 4096)) * 0.02
    image_tokens = z @ llm_proj
    print(f"Image tokens para LLM: {image_tokens.shape} (32, 4096)")
    return 0


if __name__ == "__main__":
    sys.exit(main())