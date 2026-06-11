"""
Lección: 03-multi-head-attention
Fase: 07
Multi-head: paraleliza h cabezas, cada una con su Q, K, V. Concat + W_O.
"""
from __future__ import annotations
import sys
import numpy as np


def _self_attention(Q, K, V, mask=None):
    """Scaled dot-product attention: reusamos la leccion 02."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)
    return weights @ V, weights


def split_heads(x, n_heads, d_k):
    """x: (seq, d_model) -> (n_heads, seq, d_k)."""
    seq_len, d_model = x.shape
    return x.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)


def merge_heads(x):
    """x: (n_heads, seq, d_k) -> (seq, n_heads * d_k)."""
    n_heads, seq_len, d_k = x.shape
    return x.transpose(1, 0, 2).reshape(seq_len, n_heads * d_k)


def multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads, mask=None):
    """X: (seq, d_model). Returns: (seq, d_model)."""
    seq_len, d_model = X.shape
    assert d_model % n_heads == 0
    d_k = d_model // n_heads
    # Proyecciones: (seq, d_model) @ (d_model, d_model) -> (seq, d_model)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    # Split: (n_heads, seq, d_k)
    Q_h = split_heads(Q, n_heads, d_k)
    K_h = split_heads(K, n_heads, d_k)
    V_h = split_heads(V, n_heads, d_k)
    # Attention por head: (n_heads, seq, d_k) -> (n_heads, seq, d_k)
    head_outs = []
    for h in range(n_heads):
        out_h, _ = _self_attention(Q_h[h], K_h[h], V_h[h], mask=mask)
        head_outs.append(out_h)
    heads = np.stack(head_outs)  # (n_heads, seq, d_k)
    # Concat + output projection
    concat = merge_heads(heads)  # (seq, n_heads * d_k) = (seq, d_model)
    return concat @ W_O


def init_params(d_model, seed=0):
    """Inicializacion Xavier-ish: W ~ N(0, 1/sqrt(d_model))."""
    rng = np.random.default_rng(seed)
    W = lambda: rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
    return W(), W(), W(), W()


def main() -> int:
    seq, d_model, n_heads = 4, 8, 2
    X = np.random.default_rng(42).standard_normal((seq, d_model))
    W_Q, W_K, W_V, W_O = init_params(d_model)
    out = multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads)
    print(f"input shape:  {X.shape}")
    print(f"output shape: {out.shape}")
    print(f"n_heads: {n_heads}, d_k: {d_model // n_heads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())