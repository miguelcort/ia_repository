"""
Lección: 01-por-que-transformers
Fase: 07
Prerrequisitos: 03-nucleo-deep-learning/10-mini-framework
"""
from __future__ import annotations
import sys
import numpy as np


def path_length_analysis(seq_len, model="rnn", n_layers=12):
    """Path length analysis: maxima distancia entre tokens.
    RNN: O(n). Self-attention: O(1). CNN: O(log_k(n))."""
    if model == "rnn":
        return seq_len  # O(n) path length
    elif model == "self-attention":
        return 1  # O(1)
    elif model == "cnn":
        # log_k(n) donde k es kernel size
        k = 3
        return int(np.ceil(np.log(seq_len) / np.log(k)))
    return -1


def compute_complexity(seq_len, dim, model="rnn"):
    """FLOPs per layer para una secuencia de tamano seq_len.
    RNN: O(n * d^2). Self-attention: O(n^2 * d). CNN: O(k * n * d^2)."""
    if model == "rnn":
        return seq_len * dim * dim
    elif model == "self-attention":
        return seq_len * seq_len * dim
    elif model == "cnn":
        k = 3
        return k * seq_len * dim * dim
    return -1


def main() -> int:
    n = 1000
    d = 512
    for modelo in ["rnn", "cnn", "self-attention"]:
        pl = path_length_analysis(n, modelo)
        flops = compute_complexity(n, d, modelo)
        print(f"{modelo:15s} path_length={pl:4d}, FLOPs/layer={flops:>15.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())