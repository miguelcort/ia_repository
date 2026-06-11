"""
Lección: 02-self-attention-desde-cero
Fase: 07
Self-attention con scale y softmax: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    """Numericamente estable: resta el max."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.
    Q, K, V: (seq_len, d_k). mask: (seq_len, seq_len) o None.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T  # (seq_len, seq_len)
    scores = scores / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask  # mask: -inf donde se bloquea
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def attention_weights_demo(seq_len, d_k):
    """Visualiza los pesos de attention promediando heads."""
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_k))
    out, weights = self_attention(Q, K, V)
    return weights


def causal_mask(seq_len):
    """Mask triangular inferior: token i solo atiende j <= i."""
    return np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)


def main() -> int:
    seq, d = 6, 4
    weights = attention_weights_demo(seq, d)
    print("Pesos de attention (shape, row sums):")
    print(f"  shape={weights.shape}, row sums={weights.sum(axis=-1).round(2)}")
    # Causal
    Q = K = V = np.ones((seq, d))
    out, w = self_attention(Q, K, V, mask=causal_mask(seq))
    print(f"\nCausal row sums: {w.sum(axis=-1).round(2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())