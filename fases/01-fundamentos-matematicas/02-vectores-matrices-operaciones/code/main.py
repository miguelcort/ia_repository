"""
Lección: 02-vectores-matrices-operaciones
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
Fuentes:
- NumPy: https://numpy.org/doc/stable/
"""
from __future__ import annotations

import sys

import numpy as np


def producto_punto(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vectores deben tener la misma forma")
    return float(np.sum(a * b))


def producto_exterior(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b)


def producto_hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Vectores deben tener la misma forma")
    return a * b


def norma_l2(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v ** 2)))


def coseno(a: np.ndarray, b: np.ndarray) -> float:
    na = norma_l2(a)
    nb = norma_l2(b)
    if na == 0 or nb == 0:
        return 0.0
    return producto_punto(a, b) / (na * nb)


def verificar_propiedades() -> dict:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([7.0, 8.0, 9.0])
    resultados = {}
    # Distributividad: a.(b+c) == a.b + a.c
    resultados["distributividad"] = bool(np.isclose(
        producto_punto(a, b + c),
        producto_punto(a, b) + producto_punto(a, c),
    ))
    # Simetria: a.b == b.a
    resultados["simetria_punto"] = bool(np.isclose(
        producto_punto(a, b), producto_punto(b, a),
    ))
    # No conmutatividad del producto matricial
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    resultados["matmul_no_conmutativo"] = not np.allclose(A @ B, B @ A)
    # Asociatividad del producto matricial: (AB)C == A(BC)
    C = np.array([[2.0, 0.0], [1.0, 2.0]])
    resultados["matmul_asociativo"] = bool(np.allclose((A @ B) @ C, A @ (B @ C)))
    return resultados


def main() -> int:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print("a =", a)
    print("b =", b)
    print("a . b =", producto_punto(a, b))
    print("a outer b =\n", producto_exterior(a, b))
    print("a hadamard b =", producto_hadamard(a, b))
    print("cos(a, b) =", coseno(a, b))
    print("\nPropiedades:", verificar_propiedades())
    return 0


if __name__ == "__main__":
    sys.exit(main())
