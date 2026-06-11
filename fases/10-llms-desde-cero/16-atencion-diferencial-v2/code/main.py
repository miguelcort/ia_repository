"""
Lección: 16-atencion-diferencial-v2
Fase: 10
Differential Attention (Microsoft 2024): substract attention scores.
Cancela noise, focus on signal. Usado en Phi-4.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def standard_attention(Q, K, V, mask=None):
    """Standard scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ V


def differential_attention(Q, K, V, lambda_q=1.0, lambda_k=1.0, mask=None):
    """Differential Attention (Microsoft 2024).
    DiffAttn = (Q @ K^T - lambda_q * Q' @ K'^T) / sqrt(2 * d_k).
    Donde Q', K' son rotaciones (rotary) de Q, K.
    Output = DiffAttn @ V.
    """
    d_k = Q.shape[-1]
    # Split: mitad de Q, K van al signal, otra mitad al noise
    half = d_k // 2
    Q1, Q2 = Q[:, :half], Q[:, half:]
    K1, K2 = K[:, :half], K[:, half:]
    # Compute both attentions
    s1 = Q1 @ K1.T / np.sqrt(d_k)
    s2 = Q2 @ K2.T / np.sqrt(d_k)
    # Differential: subtract noise from signal
    diff_scores = s1 - lambda_q * s2
    if mask is not None:
        diff_scores = diff_scores + mask
    weights = softmax(diff_scores, axis=-1)
    return weights @ V


def differential_components():
    return {
        "Signal-noise split": "Q, K split en mitades, sub para cancel noise",
        "Lambda": "Lambda_q, lambda_k scaling",
        "Scale": "1/sqrt(2*d_k) en vez de 1/sqrt(d_k)",
        "Output": "Diff scores @ V, mismo shape",
        "Phi-4": "Usado en Microsoft Phi-4, SOTA",
    }


def main() -> int:
    rng = np.random.default_rng(0)
    n, d = 8, 8
    Q = rng.standard_normal((n, d))
    K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))
    out_std = standard_attention(Q, K, V)
    out_diff = differential_attention(Q, K, V)
    print(f"Standard output: shape={out_std.shape}, mean={out_std.mean():.3f}")
    print(f"Diff output: shape={out_diff.shape}, mean={out_diff.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())