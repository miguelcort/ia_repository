"""
Lección: 09-programacion-de-learning-rate
Fase: 03
Prerrequisitos: 06-optimizadores
"""
from __future__ import annotations
import sys
import numpy as np


def lr_constante(paso, lr_inicial=0.1):
    return lr_inicial


def lr_step(paso, lr_inicial=0.1, drop=0.1, cada=10):
    """Step decay: cada 'cada' pasos, multiplicar lr por drop."""
    return lr_inicial * (drop ** (paso // cada))


def lr_exponential(paso, lr_inicial=0.1, gamma=0.95):
    """Exponential decay: lr = lr_inicial * gamma^paso."""
    return lr_inicial * (gamma ** paso)


def lr_cosine(paso, lr_max=0.1, lr_min=0.0, T_max=100):
    """Cosine annealing: lr sigue coseno de lr_max a lr_min en T_max pasos."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * paso / T_max))


def lr_warmup_cosine(paso, warmup=10, lr_max=0.1, lr_min=0.0, T_max=100):
    """Warmup + cosine: lr sube linealmente en warmup, luego cosine decay."""
    if paso < warmup:
        return lr_max * (paso + 1) / warmup
    return lr_cosine(paso - warmup, lr_max=lr_max, lr_min=lr_min, T_max=T_max - warmup)


def lr_reduce_on_plateau(historial, factor=0.5, paciencia=3, lr_min=1e-6):
    """Reduce lr cuando val loss no mejora. historial: lista de val losses."""
    if len(historial) <= paciencia:
        return historial[0] if historial else 0.1
    mejor = min(historial[:-paciencia])
    ultimos = historial[-paciencia:]
    if all(v >= mejor for v in ultimos):
        return max(historial[-1] * factor, lr_min)
    return historial[-1]


def main() -> int:
    n = 30
    print("paso | const  | step    | exp     | cosine  | warmup-cos")
    for p in range(0, n, 5):
        c = lr_constante(p, 0.1)
        s = lr_step(p, 0.1, 0.5, 10)
        e = lr_exponential(p, 0.1, 0.95)
        cos = lr_cosine(p, 0.1, 0.0, n)
        w = lr_warmup_cosine(p, 5, 0.1, 0.0, n)
        print(f"{p:3d}  | {c:.4f} | {s:.4f} | {e:.4f} | {cos:.4f} | {w:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())