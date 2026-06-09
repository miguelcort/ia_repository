"""
Lección: 10-reduccion-de-dimensionalidad
Fase: 01
Prerrequisitos: 03-transformaciones-valores-propios, 06-probabilidad-y-distribuciones
"""
from __future__ import annotations

import sys

import numpy as np


def centrar(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0)


def covarianza(X: np.ndarray) -> np.ndarray:
    Xc = centrar(X)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA via eigendecomposicion. Devuelve (componentes, datos_proyectados)."""
    Xc = centrar(X)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    autovalores, autovectores = np.linalg.eigh(cov)
    # Tomamos los k mayores autovalores
    idx = np.argsort(autovalores)[::-1][:k]
    componentes = autovectores[:, idx]
    return componentes, Xc @ componentes


def varianza_explicada(autovalores: np.ndarray) -> np.ndarray:
    total = autovalores.sum()
    return autovalores[::-1] / total


def main() -> int:
    rng = np.random.default_rng(42)
    # Datos 3D con correlacion: la tercera dimension es casi combinacion lineal
    n = 200
    X = rng.normal(size=(n, 3))
    X[:, 2] = X[:, 0] + X[:, 1] + 0.01 * rng.normal(size=n)
    print(f"Datos: {X.shape}")
    cov = covarianza(X)
    print(f"Covarianza:\n{cov}")
    componentes, proyectados = pca(X, k=2)
    print(f"Componentes PCA shape: {componentes.shape}")
    print(f"Proyectados shape: {proyectados.shape}")
    # Reconstruccion
    reconstruido = proyectados @ componentes.T
    error = np.mean((X - X.mean(axis=0) - reconstruido) ** 2)
    print(f"Error de reconstruccion: {error:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
