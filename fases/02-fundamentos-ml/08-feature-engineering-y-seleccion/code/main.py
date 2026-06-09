"""
Lección: 08-feature-engineering-y-seleccion
Fase: 02
Prerrequisitos: 02-regresion-lineal-desde-cero
"""
from __future__ import annotations
import sys
import numpy as np


def estandarizar(X):
    """Z-score: (X - media) / std."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    # Evitar division por cero
    sigma[sigma == 0] = 1
    return (X - mu) / sigma


def min_max(X):
    """Escala a [0, 1]."""
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    rango = mx - mn
    rango[rango == 0] = 1
    return (X - mn) / rango


def one_hot(y, n_clases=None):
    """One-hot encoding de etiquetas."""
    if n_clases is None:
        n_clases = int(y.max() + 1)
    out = np.zeros((len(y), n_clases), dtype=int)
    out[np.arange(len(y)), y] = 1
    return out


def polynomial_features(X, grado=2):
    """Features polinomicas: x, x^2, x1*x2, ..."""
    n, d = X.shape
    feats = [X]
    for g in range(2, grado + 1):
        # Generar combinaciones con repeticion
        from itertools import combinations_with_replacement
        for combo in combinations_with_replacement(range(d), g):
            col = np.ones(n)
            for idx in combo:
                col = col * X[:, idx]
            feats.append(col.reshape(-1, 1))
    return np.hstack(feats)


def main() -> int:
    X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
    print("Original:\n", X)
    print("Estandarizado:\n", estandarizar(X))
    print("Min-max:\n", min_max(X))
    y = np.array([0, 1, 2, 1])
    print("One-hot:\n", one_hot(y))
    Xp = polynomial_features(X, grado=2)
    print(f"Polinomio grado 2: shape {Xp.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())