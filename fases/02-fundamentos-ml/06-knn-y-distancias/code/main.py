"""
Lección: 06-knn-y-distancias
Fase: 02
Prerrequisitos: 05-support-vector-machines
"""
from __future__ import annotations
import sys
import numpy as np


def euclidiana(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def knn(X_train, y_train, X_test, k=3):
    """K-Nearest Neighbors: voto por mayoria entre los k mas cercanos."""
    y_pred = []
    for x in X_test:
        dists = np.array([euclidiana(x, xt) for xt in X_train])
        idx = np.argsort(dists)[:k]
        vecinos = y_train[idx]
        # Voto por mayoria
        valores, counts = np.unique(vecinos, return_counts=True)
        y_pred.append(valores[np.argmax(counts)])
    return np.array(y_pred)


def main() -> int:
    rng = np.random.default_rng(42)
    X_tr = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(3, 0.5, (30, 2))])
    y_tr = np.array([0] * 30 + [1] * 30)
    X_te = np.array([[0.0, 0.0], [3.0, 3.0]])
    y_pred = knn(X_tr, y_tr, X_te, k=3)
    print(f"Predicciones: {y_pred} (esperado [0, 1])")
    return 0


if __name__ == "__main__":
    sys.exit(main())