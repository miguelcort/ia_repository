"""
Lección: 10-mecanismo-de-atencion
Fase: 05
Prerrequisitos: 09-secuencia-a-secuencia
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x_est = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x_est)
    return exp / exp.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
    Q: (T_q, d_k), K: (T_k, d_k), V: (T_k, d_v). mask: opcional (T_q, T_k)."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    pesos = softmax(scores, axis=-1)
    return pesos @ V, pesos


def bahdanau_attention(h_dec, h_enc_list):
    """Bahdanau (additive) attention: e_t = v tanh(W_h h_dec + W_s h_enc_t).
    Score entre h_dec y cada h_enc.
    """
    # Mock: simplificado a dot-product para el ejemplo
    scores = np.array([float(h_dec @ h_e) for h_e in h_enc_list])
    pesos = softmax(scores)
    context = sum(p * h_e for p, h_e in zip(pesos, h_enc_list))
    return context, pesos


def luong_attention(h_dec, h_enc_list):
    """Luong (dot-product) attention: e_t = h_dec^T h_enc_t."""
    scores = np.array([float(h_dec @ h_e) for h_e in h_enc_list])
    pesos = softmax(scores)
    context = sum(p * h_e for p, h_e in zip(pesos, h_enc_list))
    return context, pesos


def cross_entropy(logits, target):
    return float(-np.log(logits[target] + 1e-9))


def main() -> int:
    # Scaled dot-product
    T_q, T_k, d = 4, 5, 8
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(T_q, d))
    K = rng.normal(size=(T_k, d))
    V = rng.normal(size=(T_k, d))
    out, pesos = scaled_dot_product_attention(Q, K, V)
    print(f"SDP output shape: {out.shape}, pesos shape: {pesos.shape}")
    print(f"Sum pesos por query: {pesos.sum(axis=-1).round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())