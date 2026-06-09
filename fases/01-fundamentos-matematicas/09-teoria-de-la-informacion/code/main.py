"""
Lección: 09-teoria-de-la-informacion
Fase: 01
Prerrequisitos: 06-probabilidad-y-distribuciones
"""
from __future__ import annotations

import sys
from math import log2

import numpy as np


def entropia(probs: np.ndarray, base: float = 2.0) -> float:
    """Entropia de Shannon: H = -sum p log p. Base 2 da bits."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    return float(-np.sum(p * np.log(p) / np.log(base)))


def entropia_conjunta(p_xy: np.ndarray) -> float:
    """Entropia conjunta: H(X,Y) = -sum_{x,y} p(x,y) log p(x,y)."""
    p = np.asarray(p_xy, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def entropia_condicional(p_xy: np.ndarray) -> float:
    """H(X|Y) = H(X,Y) - H(Y)."""
    p = np.asarray(p_xy, dtype=float)
    p_y = p.sum(axis=0)
    h_y = entropia(p_y)
    h_xy = entropia_conjunta(p)
    return h_xy - h_y


def informacion_mutua(p_xy: np.ndarray) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y)."""
    p = np.asarray(p_xy, dtype=float)
    p_x = p.sum(axis=1)
    p_y = p.sum(axis=0)
    h_x = entropia(p_x)
    h_y = entropia(p_y)
    h_xy = entropia_conjunta(p)
    return h_x + h_y - h_xy


def divergencia_kl(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P || Q) = sum p log(p/q). Mide cuantas bits extra
    necesitas si usas Q para codificar datos de P."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(p * np.log2(p / q)))


def entropia_cruzada(p: np.ndarray, q: np.ndarray) -> float:
    """H(P, Q) = -sum p log q. Es la perdida clasica en clasificacion."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(-np.sum(p * np.log2(q)))


def main() -> int:
    # Moneda justa
    p_justa = np.array([0.5, 0.5])
    print(f"Entropia moneda justa: {entropia(p_justa):.3f} bits (maximo 1)")
    # Moneda sesgada
    p_sesgada = np.array([0.9, 0.1])
    print(f"Entropia moneda sesgada: {entropia(p_sesgada):.3f} bits (< 1)")
    # Distribucion conjunta
    p_xy = np.array([[0.25, 0.25], [0.25, 0.25]])
    print(f"H(X,Y) si X=Y uniforme: {entropia_conjunta(p_xy):.3f}")
    # Informacion mutua
    p_xy_dep = np.array([[0.4, 0.1], [0.1, 0.4]])
    print(f"I(X;Y) si dependientes: {informacion_mutua(p_xy_dep):.3f} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
