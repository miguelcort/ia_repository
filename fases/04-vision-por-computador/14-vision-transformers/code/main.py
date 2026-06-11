"""
Lección: 14-vision-transformers
Fase: 04
Prerrequisitos: 03-cnns-desde-lenet-hasta-resnet
"""
from __future__ import annotations
import sys
import numpy as np


def dividir_en_patches(img, patch_size=16):
    """Divide imagen (H, W, C) en patches de tamano patch_size.
    Output: (n_patches, patch_size*patch_size*C)."""
    H, W, C = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    n_h = H // patch_size
    n_w = W // patch_size
    patches = img.reshape(n_h, patch_size, n_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # n_h, n_w, p, p, C
    return patches.reshape(n_h * n_w, -1)


def patch_embedding(patches, dim_embedding=768, semilla=0):
    """Proyecta cada patch a dim_embedding. Output: (n_patches, dim_embedding)."""
    rng = np.random.default_rng(semilla)
    n_patches, patch_dim = patches.shape
    W = rng.normal(scale=np.sqrt(1.0 / patch_dim), size=(patch_dim, dim_embedding))
    return patches @ W


def positional_embedding(n_patches, dim_embedding, semilla=0):
    """PE aprendido o senos/cosenos. Aqui: senos/cosenos estilo Transformer."""
    pe = np.zeros((n_patches, dim_embedding))
    for pos in range(n_patches):
        for i in range(dim_embedding):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / (10000 ** (i / dim_embedding)))
            else:
                pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / dim_embedding)))
    return pe


def self_attention_simple(x, dim_head=64):
    """Self-attention: Q, K, V proyecciones y softmax(QK^T / sqrt(d))V.
    Input: (n_tokens, dim_model). Output: (n_tokens, dim_model)."""
    n_tokens, dim = x.shape
    rng = np.random.default_rng(0)
    W_q = rng.normal(scale=0.1, size=(dim, dim_head))
    W_k = rng.normal(scale=0.1, size=(dim, dim_head))
    W_v = rng.normal(scale=0.1, size=(dim, dim_head))
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K.T / np.sqrt(dim_head)
    # Softmax
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    pesos = exp / exp.sum(axis=-1, keepdims=True)
    return pesos @ V


def classification_head(features, n_clases=10, semilla=0):
    """Cabeza de clasificacion: features (n_tokens, dim) -> (n_clases,)."""
    rng = np.random.default_rng(semilla)
    # Pool: promedio de todos los tokens
    pooled = features.mean(axis=0)
    W = rng.normal(scale=0.1, size=(pooled.shape[0], n_clases))
    return pooled @ W


def main() -> int:
    img = np.random.default_rng(0).normal(size=(224, 224, 3))
    patches = dividir_en_patches(img, patch_size=16)
    print(f"Patches: {patches.shape}")  # (196, 768)
    # +1 para cls token
    emb = patch_embedding(patches)
    pe = positional_embedding(patches.shape[0] + 1, 768)
    print(f"PE shape: {pe.shape}")
    # Anade CLS token (simplificado)
    cls = np.zeros((1, 768))
    x = np.vstack([cls, emb]) + pe
    out = self_attention_simple(x)
    print(f"Attn output: {out.shape}")
    logits = classification_head(out)
    print(f"Logits: {logits.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())