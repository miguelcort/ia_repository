"""
Lección: 15-variantes-de-atencion
Fase: 07
Variantes: linear attention, sparse (longformer), local-global sliding window.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def standard_attention(Q, K, V, mask=None):
    """O(n^2) memoria."""
    d_k = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        s = s + mask
    w = softmax(s, axis=-1)
    return w @ V, w


def linear_attention(Q, K, V, kernel="elu", eps=1e-6):
    """Linear attention O(n) memoria: kernel feature map.
    phi(x) = elu(x) + 1 (no negatividad) o ReLU.
    """
    if kernel == "elu":
        phi = lambda x: np.maximum(x, 0) + 1  # approx elu+1
    elif kernel == "relu":
        phi = lambda x: np.maximum(x, 0)
    else:
        phi = lambda x: x
    Q_p = phi(Q)
    K_p = phi(K)
    # O(n * d_k^2) en vez de O(n^2 * d_k)
    # Q @ (K^T @ V): (n, d) @ ((d, d) resultado) = (n, d)
    KV = K_p.T @ V  # (d, d)
    out = (Q_p @ KV) / (Q_p @ K_p.sum(axis=0).reshape(-1, 1) + eps)
    return out, None  # weights no son n x n


def sliding_window_attention(Q, K, V, window=3):
    """Atencion local: cada token ve solo window vecinos."""
    seq = Q.shape[0]
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Mask: -inf fuera de ventana
    mask = np.ones((seq, seq)) * -1e9
    for i in range(seq):
        lo = max(0, i - window)
        hi = min(seq, i + window + 1)
        mask[i, lo:hi] = 0
    scores = scores + mask
    w = softmax(scores, axis=-1)
    return w @ V, w


def global_local_attention(Q, K, V, n_global=2, window=2):
    """Longformer-style: primeros n_global tokens son globales, resto sliding window."""
    seq = Q.shape[0]
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.ones((seq, seq)) * -1e9
    # Globales
    for i in range(n_global):
        mask[i, :] = 0  # global token ve todo
        mask[:, i] = 0  # todos ven global token
    # Sliding window para resto
    for i in range(n_global, seq):
        lo = max(0, i - window)
        hi = min(seq, i + window + 1)
        mask[i, lo:hi] = 0
    scores = scores + mask
    w = softmax(scores, axis=-1)
    return w @ V, w


def memory_comparison(seq_len):
    """Memoria (bytes) para diferentes variantes."""
    n = seq_len
    return {
        "standard": n * n * 2,  # fp16
        "linear": n * 2,  # O(n)
        "sliding_window": n * 2 * window,  # O(n*window)
    }


def main() -> int:
    n, d = 8, 4
    Q = np.random.default_rng(0).standard_normal((n, d))
    K = np.random.default_rng(1).standard_normal((n, d))
    V = np.random.default_rng(2).standard_normal((n, d))
    out, _ = standard_attention(Q, K, V)
    print(f"Standard attention: {out.shape}")
    out_l, _ = linear_attention(Q, K, V, kernel="elu")
    print(f"Linear attention: {out_l.shape}")
    out_sw, _ = sliding_window_attention(Q, K, V, window=2)
    print(f"Sliding window (w=2): {out_sw.shape}")
    out_gl, _ = global_local_attention(Q, K, V, n_global=1, window=2)
    print(f"Global-local: {out_gl.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())