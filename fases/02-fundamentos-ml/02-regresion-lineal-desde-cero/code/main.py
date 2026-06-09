"""
Lección: 02-regresion-lineal-desde-cero
Fase: 02
Prerrequisitos: 01-que-es-machine-learning
"""
from __future__ import annotations
import sys
import numpy as np


def ajustar(X, y):
    """Regresion lineal por minimos cuadrados. y = X @ w + b."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # Anadir columna de unos para el intercepto
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    # Resolver (X^T X) w = X^T y
    w = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)
    return w


def predecir(w, X):
    X = np.asarray(X, dtype=float)
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return X_aug @ w


def mse(y_verdadero, y_predicho):
    return float(np.mean((y_verdadero - y_predicho) ** 2))


def r2(y_verdadero, y_predicho):
    ss_res = np.sum((y_verdadero - y_predicho) ** 2)
    ss_tot = np.sum((y_verdadero - np.mean(y_verdadero)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def main() -> int:
    rng = np.random.default_rng(42)
    X = np.linspace(0, 5, 50)
    y = 2.5 * X + 0.7 + 0.2 * rng.normal(size=50)
    w = ajustar(X, y)
    print(f"w = {w} (esperado [0.7, 2.5])")
    y_pred = predecir(w, X)
    print(f"MSE: {mse(y, y_pred):.4f}, R^2: {r2(y, y_pred):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())