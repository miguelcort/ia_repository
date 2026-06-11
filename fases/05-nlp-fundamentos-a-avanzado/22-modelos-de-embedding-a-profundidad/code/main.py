"""
Lección: 22-modelos-de-embedding-a-profundidad
Fase: 05
Prerrequisitos: 21-inferencia-de-texto-nli
"""
from __future__ import annotations
import sys
import numpy as np


def contrastive_loss(z1, z2, t=0.07):
    """InfoNCE: maximizar similitud entre pares positivos, minimizar con negativos.
    z1, z2: (N, D) embeddings normalizados.
    """
    n = len(z1)
    z = np.vstack([z1, z2])
    sim = z @ z.T / t
    # Mascara diagonal
    etiquetas = np.concatenate([np.arange(n), np.arange(n)])
    exp = np.exp(sim - sim.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    loss = 0.0
    for i in range(n):
        pos = i + n
        loss -= np.log(probs[i, pos] + 1e-10)
        loss -= np.log(probs[pos, i] + 1e-10)
    return loss / (2 * n)


def l2_normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-9)


def mean_pool(embeddings, mask):
    """Mean pooling ignorando padding. mask: (N, T) con 1 donde hay token real."""
    mask = mask[:, :, None]  # (N, T, 1)
    summed = (embeddings * mask).sum(axis=1)
    counts = mask.sum(axis=1) + 1e-9
    return summed / counts


def cls_pool(embeddings):
    """CLS pooling: tomar el primer token (BERT-style)."""
    return embeddings[:, 0, :]


def hard_negative_mining(emb_anchor, emb_pos, emb_negs, n_hard=2):
    """Selecciona los n_hard negativos mas similares al anchor.
    Mejora la calidad de los embeddings.
    emb_anchor: (N, D). emb_negs: (N, M, D). Devuelve (N, n_hard).
    """
    N, M, D = emb_negs.shape
    # Similitud anchor vs negativos: (N, 1, D) @ (N, D, M) -> (N, 1, M) -> (N, M)
    sims = np.einsum("nd,nmd->nm", emb_anchor, emb_negs)
    # Top-n_hard negativos (mayor similitud)
    hard_idx = np.argsort(sims, axis=-1)[:, -n_hard:]
    return hard_idx


def main() -> int:
    rng = np.random.default_rng(0)
    z1 = l2_normalize(rng.normal(size=(4, 64)))
    z2 = l2_normalize(rng.normal(size=(4, 64)))
    loss = contrastive_loss(z1, z2)
    print(f"InfoNCE loss: {loss:.3f}")
    embeddings = rng.normal(size=(2, 5, 64))
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    pooled = mean_pool(embeddings, mask)
    print(f"Mean pool shape: {pooled.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())