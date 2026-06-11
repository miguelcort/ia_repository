"""
Lección: 05-transformer-completo
Fase: 07
Encoder block + Decoder block. Add+Norm, FFN, residual connections.
"""
from __future__ import annotations
import sys
import numpy as np
import math


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    """LayerNorm sobre la ultima dimension."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    """Gaussian Error Linear Unit: 0.5 * x * (1 + erf(x/sqrt(2)))."""
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))


def ffn(x, W1, b1, W2, b2):
    """Feed-forward: Linear -> GELU -> Linear."""
    return (gelu(x @ W1 + b1) @ W2 + b2)


def _attn(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    w = softmax(scores, axis=-1)
    return w @ V, w


def encoder_block(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2):
    """Pre-LN encoder: y = x + sublayer(LN(x))."""
    a, _ = _attn(x @ W_Q, x @ W_K, x @ W_V)
    h = layer_norm(x + a @ W_O)
    return layer_norm(h + ffn(h, W1, b1, W2, b2))


def decoder_block(x, enc_out, W_Q, W_K, W_V, W_O,
                  W_Q2, W_K2, W_V2, W_O2, W1, b1, W2, b2,
                  causal_mask):
    """Decoder: masked self-attn -> cross-attn -> FFN."""
    # Self-attn causal
    a, _ = _attn(x @ W_Q, x @ W_K, x @ W_V, mask=causal_mask)
    h = layer_norm(x + a @ W_O)
    # Cross-attn: Q de decoder, K/V de encoder
    a2, _ = _attn(h @ W_Q2, enc_out @ W_K2, enc_out @ W_V2)
    h = layer_norm(h + a2 @ W_O2)
    return layer_norm(h + ffn(h, W1, b1, W2, b2))


def init_block(d, d_ff, seed=0):
    """Inicializa pesos para un bloque."""
    rng = np.random.default_rng(seed)
    s = lambda: 1.0 / np.sqrt(d)
    return (s() * rng.standard_normal((d, d)),  # W_Q
            s() * rng.standard_normal((d, d)),  # W_K
            s() * rng.standard_normal((d, d)),  # W_V
            s() * rng.standard_normal((d, d)),  # W_O
            s() * rng.standard_normal((d, d_ff)),  # W1
            np.zeros(d_ff),  # b1
            s() * rng.standard_normal((d_ff, d)),  # W2
            np.zeros(d))  # b2


def main() -> int:
    seq, d, d_ff = 4, 8, 32
    x = np.random.default_rng(0).standard_normal((seq, d))
    enc = np.random.default_rng(1).standard_normal((seq, d))
    weights = init_block(d, d_ff, seed=42)
    out = encoder_block(x, *weights)
    print(f"Encoder block: input {x.shape} -> output {out.shape}")
    W_Q, W_K, W_V, W_O, W1, b1, W2, b2 = weights
    out2 = encoder_block(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2)
    # Re-init decoder
    dec_weights = init_block(d, d_ff, seed=43)
    cm = np.triu(np.ones((seq, seq)) * -1e9, k=1)
    d_out = decoder_block(x, enc, *dec_weights[:4],
                          *dec_weights[:4], *dec_weights[4:8], cm)
    print(f"Decoder block: input {x.shape} -> output {d_out.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())