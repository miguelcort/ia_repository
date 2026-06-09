"""
Lección: 07-aprendizaje-no-supervisado
Fase: 02
Prerrequisitos: 06-knn-y-distancias
"""
from __future__ import annotations
import sys
import numpy as np


def kmeans(X, k, max_iter=100, semilla=0):
    """K-Means clustering. Devuelve (centroides, asignaciones)."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    centroides = X[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(max_iter):
        # Asignar cada punto al centroide mas cercano
        dists = np.array([[np.linalg.norm(x - c) for c in centroides] for x in X])
        asignaciones = np.argmin(dists, axis=1)
        # Recalcular centroides
        nuevos = np.array([X[asignaciones == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroides, nuevos, atol=1e-6):
            break
        centroides = nuevos
    return centroides, asignaciones


def inercia(X, asignaciones, centroides):
    """Suma de distancias cuadradas intra-cluster."""
    total = 0.0
    for i, c in enumerate(centroides):
        puntos = X[asignaciones == i]
        total += np.sum((puntos - c) ** 2)
    return float(total)


def main() -> int:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (30, 2)), rng.normal(5, 0.5, (30, 2))])
    centroides, asig = kmeans(X, k=2, semilla=42)
    print(f"Centroides:\n{centroides}")
    print(f"Inercia: {inercia(X, asig, centroides):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())