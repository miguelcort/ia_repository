"""
Lección: 13-estabilidad-numerica
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
"""
from __future__ import annotations
import sys
import numpy as np


def epsilon_maquina():
    return np.finfo(float).eps


def max_min():
    f = np.finfo(float)
    return f.max, f.min


def condicion_matriz(A):
    return np.linalg.cond(A)


def softmax_inestable(x):
    """Softmax ingenuo: exp puede overflow."""
    ex = np.exp(x)
    return ex / ex.sum()


def softmax_estable(x):
    """Softmax numericamente estable: resta el max."""
    x_shift = x - np.max(x)
    ex = np.exp(x_shift)
    return ex / ex.sum()


def main() -> int:
    eps = epsilon_maquina()
    mx, mn = max_min()
    print(f"Epsilon maquina: {eps}")
    print(f"Max float64: {mx:.2e}")
    print(f"Min float64: {mn:.2e}")

    # Softmax estable vs inestable
    x = np.array([1000.0, 1001.0, 1002.0])
    try:
        inestable = softmax_inestable(x.copy())
        print(f"Softmap inestable: {inestable}")
    except Exception as e:
        print(f"Softmax inestable fallo: {e}")
    estable = softmax_estable(x)
    print(f"Softmap estable: {estable}")

    # Condicion de matriz
    A = np.array([[1.0, 1.0], [1.0, 1.0001]])
    print(f"Cond(A) casi-singular: {condicion_matriz(A):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())