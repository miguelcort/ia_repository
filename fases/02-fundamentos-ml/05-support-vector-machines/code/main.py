"""
Lección: 05-support-vector-machines
Fase: 02
Prerrequisitos: 03-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


def kernel_lineal(x, y):
    return float(np.dot(x, y))


def kernel_rbf(x, y, gamma=1.0):
    diff = np.asarray(x) - np.asarray(y)
    return float(np.exp(-gamma * np.dot(diff, diff)))


def margen(X, y, w):
    """El margen geometrico del clasificador lineal w."""
    return float(np.min(y * (X @ w) / np.linalg.norm(w)))


def hinge_loss(y, scores, reg=1.0):
    """SVM primal: (1/n) sum max(0, 1 - y_i f_i) + reg ||w||^2/2."""
    margins = np.maximum(0, 1 - y * scores)
    return float(np.mean(margins) + 0.5 * reg * np.dot(scores, scores))


def main() -> int:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.5, (20, 2)), rng.normal(1, 0.5, (20, 2))])
    y = np.array([-1] * 20 + [1] * 20)
    w = np.array([1.0, 1.0])  # sin intercepto
    X_aug = np.column_stack([np.ones(len(X)), X])
    scores = X_aug @ np.array([0.0, 1.0, 1.0])
    print(f"Margen: {margen(X, y, w):.4f}")
    print(f"Hinge loss: {hinge_loss(y, scores):.4f}")
    print(f"Kernel RBF(0,0): {kernel_rbf(np.zeros(2), np.zeros(2))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())