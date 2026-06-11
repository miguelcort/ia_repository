"""
Lección: 04-positional-encoding
Fase: 07
Sinusoidal PE (original Transformer) y rotary (RoPE, usado en Llama, Mistral).
"""
from __future__ import annotations
import sys
import numpy as np


def sinusoidal_pe(seq_len, d_model):
    """PE(pos, 2i) = sin(pos / 10000^{2i/d}). PE(pos, 2i+1) = cos(...)."""
    pos = np.arange(seq_len).reshape(-1, 1)  # (seq, 1)
    i = np.arange(d_model).reshape(1, -1)  # (1, d)
    angle = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def rope(q, k, base=10000):
    """Rotary Position Embedding.
    q, k: (seq, d). Aplica rotacion a pares de dimensiones.
    """
    seq, d = q.shape
    assert d % 2 == 0
    pos = np.arange(seq).reshape(-1, 1)  # (seq, 1)
    i = np.arange(0, d, 2).reshape(1, -1)  # (1, d/2)
    theta = pos / (base ** (2 * i / d))  # (seq, d/2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # q: (seq, d/2, 2) -> split en pares
    q_pairs = q.reshape(seq, d // 2, 2)
    k_pairs = k.reshape(seq, d // 2, 2)
    # Rotacion por theta: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
    q_rot = np.empty_like(q_pairs)
    q_rot[..., 0] = q_pairs[..., 0] * cos_t - q_pairs[..., 1] * sin_t
    q_rot[..., 1] = q_pairs[..., 0] * sin_t + q_pairs[..., 1] * cos_t
    k_rot = np.empty_like(k_pairs)
    k_rot[..., 0] = k_pairs[..., 0] * cos_t - k_pairs[..., 1] * sin_t
    k_rot[..., 1] = k_pairs[..., 0] * sin_t + k_pairs[..., 1] * cos_t
    return q_rot.reshape(seq, d), k_rot.reshape(seq, d)


def alibi_bias(seq_len, n_heads):
    """ALiBi: bias aditivo lineal por distancia. slope = 2^(-8/n * h)."""
    slopes = 2 ** (-(8 / n_heads) * np.arange(1, n_heads + 1))
    pos = np.arange(seq_len)
    rel = pos.reshape(-1, 1) - pos.reshape(1, -1)  # (seq, seq)
    rel = -np.abs(rel)  # distancias negativas
    return slopes.reshape(-1, 1, 1) * rel.reshape(1, seq_len, seq_len)


def main() -> int:
    seq, d, h = 8, 16, 4
    pe = sinusoidal_pe(seq, d)
    print(f"Sinusoidal PE: shape={pe.shape}, range=[{pe.min():.3f}, {pe.max():.3f}]")
    q = np.random.default_rng(0).standard_normal((seq, d))
    k = np.random.default_rng(1).standard_normal((seq, d))
    qr, kr = rope(q, k)
    print(f"RoPE: q shape preserved: {qr.shape == q.shape}")
    bias = alibi_bias(seq, h)
    print(f"ALiBi bias: shape={bias.shape}, max={bias.max():.3f}, min={bias.min():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())