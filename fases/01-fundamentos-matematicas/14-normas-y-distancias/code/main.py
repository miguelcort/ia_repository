"""
Lección: 14-normas-y-distancias
Fase: 01
Prerrequisitos: 12-operaciones-con-tensores
"""
from __future__ import annotations
import sys
import numpy as np


def norma_l1(v):
    return float(np.sum(np.abs(v)))


def norma_l2(v):
    return float(np.sqrt(np.sum(v ** 2)))


def norma_lp(v, p):
    return float((np.sum(np.abs(v) ** p)) ** (1 / p))


def norma_linf(v):
    return float(np.max(np.abs(v)))


def distancia_euclidiana(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def distancia_manhattan(a, b):
    return float(np.sum(np.abs(a - b)))


def distancia_coseno(a, b):
    return 1.0 - float(np.dot(a, b) / (norma_l2(a) * norma_l2(b)))


def main() -> int:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    print(f"||a||_1 = {norma_l1(a)}")
    print(f"||a||_2 = {norma_l2(a):.4f}")
    print(f"||a||_inf = {norma_linf(a)}")
    print(f"d_euclidiana(a,b) = {distancia_euclidiana(a, b):.4f}")
    print(f"d_manhattan(a,b) = {distancia_manhattan(a, b):.4f}")
    print(f"d_coseno(a,b) = {distancia_coseno(a, b):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())