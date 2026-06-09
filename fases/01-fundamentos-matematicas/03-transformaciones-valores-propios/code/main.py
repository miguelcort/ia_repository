"""
Lección: 03-transformaciones-valores-propios
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
Fuentes:
- NumPy linalg: https://numpy.org/doc/stable/reference/routines.linalg.html
"""
from __future__ import annotations

import sys

import numpy as np


def matriz_rotacion(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def matriz_escalado(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0], [0.0, sy]])


def aplicar_transformacion(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    return T @ v


def valores_propios(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula valores y vectores propios. A v = lambda v."""
    return np.linalg.eig(A)


def verificar_vpropios(A: np.ndarray, tol: float = 1e-9) -> bool:
    autovalores, autovectores = valores_propios(A)
    for i in range(A.shape[0]):
        v = autovectores[:, i]
        lam = autovalores[i]
        residual = np.linalg.norm(A @ v - lam * v)
        if residual > tol:
            return False
    return True


def diagonalizar(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (P, D, P^-1) tal que A = P @ D @ P^-1."""
    autovalores, P = valores_propios(A)
    D = np.diag(autovalores.astype(float))
    P_inv = np.linalg.inv(P)
    return P, D, P_inv


def main() -> int:
    # Transformacion: rotacion 45 grados
    R = matriz_rotacion(np.pi / 4)
    v = np.array([1.0, 0.0])
    print("R =\n", R)
    print("v =", v)
    print("R @ v =", aplicar_transformacion(R, v))
    print()
    # Autovalores
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    autovalores, autovectores = valores_propios(A)
    print("A =\n", A)
    print("autovalores =", autovalores)
    print("autovectores =\n", autovectores)
    print("A v = lambda v?", verificar_vpropios(A))
    return 0


if __name__ == "__main__":
    sys.exit(main())
