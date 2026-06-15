"""
Lección: 32-token-positional-embeddings
Fase: 19
Capstone de ingeniería AI: 32 Token Positional Embeddings.
"""
from __future__ import annotations
import sys

import numpy as np


def sinusoidal_pos_emb(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / 10000 ** (i / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
