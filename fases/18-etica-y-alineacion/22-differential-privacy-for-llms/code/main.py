"""
Lección: 22-differential-privacy-for-llms
Fase: 18
Ética y alineación: 22 Differential Privacy For Llms.
"""
from __future__ import annotations
import sys
import numpy as np

import numpy as np


def clip_gradients(grads, clip_norm=1.0):
    norm = np.linalg.norm(grads)
    if norm > clip_norm:
        grads = grads * clip_norm / norm
    return grads


def add_noise(grads, sigma, clip_norm=1.0):
    noise = np.random.normal(0, sigma * clip_norm, grads.shape)
    return grads + noise


def dp_sgd_step(grads, lr=0.01, clip_norm=1.0, sigma=1.0):
    clipped = np.array([clip_gradients(g, clip_norm) for g in grads])
    noisy = add_noise(clipped.mean(axis=0), sigma, clip_norm)
    return -lr * noisy



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 22-differential-privacy-for-llms ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['clip_gradients', 'add_noise', 'dp_sgd_step']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
