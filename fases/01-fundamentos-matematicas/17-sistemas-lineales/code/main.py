"""
Lección: 17-sistemas-lineales
Fase: 01
Prerrequisitos: 11-descomposicion-svd
"""
from __future__ import annotations
import sys
import numpy as np


def resolver(A, b):
    """Resuelve Ax = b. Lanza error si A es singular."""
    return np.linalg.solve(A, b)


def lstsq(A, b):
    """Minimos cuadrados: min ||Ax - b||^2."""
    x, residuos, rango, sv = np.linalg.lstsq(A, b, rcond=None)
    return x


def condicion(A):
    return float(np.linalg.cond(A))


def main() -> int:
    # Sistema bien condicionado
    A = np.array([[3.0, 2.0], [1.0, 2.0]])
    b = np.array([7.0, 5.0])
    x = resolver(A, b)
    print(f"x = {x} (Ax={A @ x}, b={b})")
    # Sistema sobredeterminado: minimos cuadrados
    A2 = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b2 = np.array([1.0, 2.0, 2.0])
    x2 = lstsq(A2, b2)
    print(f"LSTSQ x = {x2}")
    # Condicion
    A3 = np.array([[1.0, 1.0], [1.0, 1.0001]])
    print(f"cond(A casi-singular) = {condicion(A3):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())