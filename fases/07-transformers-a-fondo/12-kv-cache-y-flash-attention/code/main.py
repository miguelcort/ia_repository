"""
Lección: 12-kv-cache-y-flash-attention
Fase: 07
KV cache: reusa K, V pasados en inference autoregresiva. Flash Attention: memoria O(n).
"""
from __future__ import annotations
import sys
import numpy as np


class KVCache:
    """Cache de K, V por capa y head."""

    def __init__(self, n_layers, n_heads, max_seq, d_k):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.d_k = d_k
        # K, V por layer: (max_seq, n_heads, d_k)
        self.k = [np.zeros((max_seq, n_heads, d_k)) for _ in range(n_layers)]
        self.v = [np.zeros((max_seq, n_heads, d_k)) for _ in range(n_layers)]
        self.cur_len = 0

    def update(self, layer, k_new, v_new):
        """k_new, v_new: (seq_new, n_heads, d_k). seq_new = 1 en inference."""
        seq_new = k_new.shape[0]
        assert self.cur_len + seq_new <= self.max_seq
        self.k[layer][self.cur_len:self.cur_len + seq_new] = k_new
        self.v[layer][self.cur_len:self.cur_len + seq_new] = v_new

    def get(self, layer):
        """Returns K, V cacheados: (cur_len, n_heads, d_k)."""
        return self.k[layer][:self.cur_len], self.v[layer][:self.cur_len]

    def advance(self, n):
        self.cur_len += n


def flash_attention_demo(Q, K, V, block_size=64):
    """Mock de Flash Attention: computa attention en bloques.
    Diferencia vs naive: no materializa scores n x n, calcula en bloques.
    """
    n = Q.shape[0]
    d_k = Q.shape[1]
    # Output acumulado (running stats estilo online softmax)
    O = np.zeros((n, d_k))
    m_i = -np.inf * np.ones(n)  # max hasta ahora
    l_i = np.zeros(n)  # denominator hasta ahora
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        # Bloque de K, V
        Kb = K[start:end]
        Vb = V[start:end]
        # Scores para tokens en [start, end)
        S = Q @ Kb.T / np.sqrt(d_k)  # (n, block)
        # Update running max, sum, output
        m_new = np.maximum(m_i, S.max(axis=-1))
        # P_ij = exp(S_ij - m_new)
        P = np.exp(S - m_new.reshape(-1, 1))
        # l_new = l_i * exp(m_i - m_new) + sum(P)
        l_new = l_i * np.exp(m_i - m_new) + P.sum(axis=-1)
        # O_new = (l_i * exp(m_i - m_new) * O + P @ Vb) / l_new
        # Solo update tokens en [start, end) (o todos si simplificamos)
        O = (l_i * np.exp(m_i - m_new)).reshape(-1, 1) * O
        # Sumamos contribution de este bloque solo a tokens relevantes
        # Simplificacion: cada token i solo ve K/V anteriores, no futuros (causal)
        # Aqui asumimos bidireccional
        O = O + P @ Vb
        O = O / l_new.reshape(-1, 1)
        m_i = m_new
        l_i = l_new
    return O


def naive_attention(Q, K, V):
    """Self-attention O(n^2) memoria, para comparar."""
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)
    P = np.exp(S - S.max(axis=-1, keepdims=True))
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def memory_comparison(seq_len, dtype_bytes=2):
    """Memoria (bytes) para naive vs flash attention."""
    n = seq_len
    naive_scores = n * n * dtype_bytes  # (n, n) en fp16/bf16
    flash_block = 64 * 64 * dtype_bytes  # block_size^2
    return naive_scores, flash_block


def main() -> int:
    cache = KVCache(n_layers=2, n_heads=4, max_seq=128, d_k=16)
    print(f"Cache: {cache.n_layers} layers, {cache.n_heads} heads, max_seq={cache.max_seq}")
    # Simular 3 tokens
    for t in range(3):
        k_new = np.random.default_rng(t).standard_normal((1, 4, 16))
        v_new = np.random.default_rng(100 + t).standard_normal((1, 4, 16))
        cache.update(layer=0, k_new=k_new, v_new=v_new)
        cache.advance(1)
    k_cached, v_cached = cache.get(layer=0)
    print(f"Despues de 3 tokens: K cache shape {k_cached.shape}")
    # Flash vs naive
    n = 64
    Q = np.random.default_rng(0).standard_normal((n, 16))
    K = np.random.default_rng(1).standard_normal((n, 16))
    V = np.random.default_rng(2).standard_normal((n, 16))
    out_naive = naive_attention(Q, K, V)
    out_flash = flash_attention_demo(Q, K, V)
    diff = np.abs(out_naive - out_flash).max()
    print(f"Diferencia naive vs flash: {diff:.2e}")
    naive_mem, flash_mem = memory_comparison(n)
    print(f"Memoria naive: {naive_mem} bytes, flash block: {flash_mem} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())