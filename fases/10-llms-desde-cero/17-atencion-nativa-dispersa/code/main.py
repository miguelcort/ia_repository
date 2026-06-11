"""
Lección: 17-atencion-nativa-dispersa
Fase: 10
Native Sparse Attention (NSA, DeepSeek 2025): sparse + compressed attention.
12x speedup, +quality en long context.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def dense_attention(Q, K, V):
    """O(n^2) memory y compute."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V


def sliding_window_attention(Q, K, V, window=64):
    """O(n*w) memory y compute. Mistral style."""
    n = Q.shape[0]
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.ones((n, n)) * -1e9
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        mask[i, lo:hi] = 0
    scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ V


def compressed_attention(Q, K, V, block_size=4):
    """Compress K, V en blocks, attention entre Q y compressed KV.
    Reduccion n -> n/block_size.
    """
    n = Q.shape[0]
    d_k = Q.shape[1]
    d_v = V.shape[1]
    n_blocks = n // block_size
    # Compress: average each block
    K_compressed = K[:n_blocks * block_size].reshape(n_blocks, block_size, d_k).mean(axis=1)
    V_compressed = V[:n_blocks * block_size].reshape(n_blocks, block_size, d_v).mean(axis=1)
    # Q attends to compressed
    scores = Q @ K_compressed.T / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    return weights @ V_compressed


def nsa_components():
    """Componentes de NSA."""
    return {
        "Compression": "Block-wise compression de K, V",
        "Selection": "Top-k blocks por importance",
        "Sliding window": "Local attention para context local",
        "Combination": "3 branches (compression + selection + sliding)",
        "Hardware": "Triton kernels, +efficient",
        "Speedup": "12x en long context vs full attention",
    }


def memory_complexity(seq_len, kind="dense", window=64, block_size=4):
    """Memoria O(...) por tipo."""
    if kind == "dense":
        return seq_len * seq_len
    elif kind == "sliding":
        return seq_len * window
    elif kind == "compressed":
        return seq_len * (seq_len // block_size)
    elif kind == "nsa":
        # NSA: O(l * r) donde l = num selected, r = compressed
        return seq_len * (seq_len // 16)  # approximate
    return 0


def main() -> int:
    n = 1024
    print(f"Dense attn mem: {memory_complexity(n, 'dense')} ({n*n})")
    print(f"Sliding window: {memory_complexity(n, 'sliding', window=64)}")
    print(f"Compressed: {memory_complexity(n, 'compressed', block_size=4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())