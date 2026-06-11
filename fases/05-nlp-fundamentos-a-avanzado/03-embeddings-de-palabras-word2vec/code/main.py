"""
Lección: 03-embeddings-de-palabras-word2vec
Fase: 05
Prerrequisitos: 02-bolsa-de-palabras-y-tfidf
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(z):
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)


def skip_gram_step(target_emb, context_emb, vocab_size, n_neg=5, semilla=0):
    """Un paso simplificado de Skip-gram con negative sampling.
    target_emb: (D,) embedding del target.
    context_emb: (D,) embedding del contexto positivo (no logits!).
    Devuelve loss.
    """
    rng = np.random.default_rng(semilla)
    # Positivo: similitud -> sigmoid -> 1
    pos_score = float(target_emb @ context_emb)
    p_pos = sigmoid(pos_score)
    loss_pos = -np.log(p_pos + 1e-10)
    # Negativos: n_neg vectores aleatorios
    neg_emb = rng.normal(0, 0.1, size=(n_neg, target_emb.shape[0]))
    p_neg = sigmoid(neg_emb @ target_emb)
    loss_neg = -np.log(1 - p_neg + 1e-10).sum()
    return float(loss_pos + loss_neg)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def analogia(emb_a, emb_b, emb_c):
    """king - man + woman ~= queen. Devuelve vec_c + vec_b - vec_a."""
    return emb_c + emb_b - emb_a


def main() -> int:
    V, D = 1000, 50
    rng = np.random.default_rng(0)
    W_in = rng.normal(0, 0.1, size=(V, D))
    target_idx = 5
    target_emb = W_in[target_idx]
    context_emb = rng.normal(0, 0.1, size=D)
    loss = skip_gram_step(target_emb, context_emb, V, n_neg=5, semilla=42)
    print(f"Loss skip-gram step: {loss:.3f}")
    # Analogia
    a, b, c = rng.normal(0, 0.1, size=(3, D))
    res = analogia(a, b, c)
    print(f"Analogia: {res.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())