"""
Lección: 04-arboles-de-decision-random-forest
Fase: 02
Prerrequisitos: 01-que-es-machine-learning
"""
from __future__ import annotations
import sys
import numpy as np


def entropia(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def ganancia_informacion(y, indices_izq, indices_der):
    n = len(y)
    n_izq, n_der = len(indices_izq), len(indices_der)
    if n_izq == 0 or n_der == 0:
        return 0.0
    h_total = entropia(y)
    h_izq = entropia(y[indices_izq])
    h_der = entropia(y[indices_der])
    h_ponderada = (n_izq * h_izq + n_der * h_der) / n
    return h_total - h_ponderada


def mejor_split(X, y):
    """Encuentra la mejor feature y umbral por ganancia de informacion."""
    n, d = X.shape
    mejor_gan = -np.inf
    for f in range(d):
        valores = np.unique(X[:, f])
        for t in valores:
            mask = X[:, f] <= t
            gan = ganancia_informacion(y, np.where(mask)[0], np.where(~mask)[0])
            if gan > mejor_gan:
                mejor_gan = gan
                mejor = (f, t)
    return mejor


def main() -> int:
    rng = np.random.default_rng(0)
    # 2 features, clases separables
    X = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(3, 1, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    f, t = mejor_split(X, y)
    print(f"Mejor split: feature={f}, umbral={t:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())