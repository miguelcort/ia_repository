"""
Lección: 01-intuicion-algebra-lineal
Fase: 01
Prerrequisitos: 00-configuracion-y-herramientas
Fuentes:
- NumPy: https://numpy.org/doc/stable/
- 3Blue1Brown "Essence of Linear Algebra": https://www.3blue1brown.com/topics/linear-algebra
"""
from __future__ import annotations

import sys

import numpy as np


def sumar(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.shape != B.shape:
        raise ValueError(f"Formas incompatibles: {A.shape} vs {B.shape}")
    return A + B


def escalar(alpha: float, A: np.ndarray) -> np.ndarray:
    return alpha * A


def transponer(A: np.ndarray) -> np.ndarray:
    return A.T


def multiplicar(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Ambos deben ser matrices 2D")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimensiones incompatibles: {A.shape} @ {B.shape}")
    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C[i, j] = sum(A[i, k] * B[k, j] for k in range(n))
    return C


def norma(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v ** 2)))


def producto_punto(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectores deben tener la misma forma")
    return float(np.sum(a * b))


def main() -> int:
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    print("A =\n", A)
    print("B =\n", B)
    print("A + B =\n", sumar(A, B))
    print("3 * A =\n", escalar(3.0, A))
    print("A @ B (manual) =\n", multiplicar(A, B))
    print("A @ B (numpy)   =\n", A @ B)
    print("A^T =\n", transponer(A))
    v = np.array([3.0, 4.0])
    print(f"||v|| = {norma(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
