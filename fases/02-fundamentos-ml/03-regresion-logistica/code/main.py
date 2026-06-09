"""
Lección: 03-regresion-logistica
Fase: 02
Prerrequisitos: 02-regresion-lineal-desde-cero
"""
from __future__ import annotations
import sys
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ajustar(X, y, lr=0.1, epochs=1000):
    """Regresion logistica por descenso por gradiente."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d + 1)  # +1 para intercepto
    X_aug = np.column_stack([np.ones(n), X])
    for _ in range(epochs):
        z = X_aug @ w
        p = sigmoid(z)
        grad = X_aug.T @ (p - y) / n
        w -= lr * grad
    return w


def predecir(w, X, umbral=0.5):
    X = np.asarray(X, dtype=float)
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return (sigmoid(X_aug @ w) >= umbral).astype(int)


def predecir_proba(w, X):
    X = np.asarray(X, dtype=float)
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return sigmoid(X_aug @ w)


def accuracy(y_verdadero, y_predicho):
    return float(np.mean(y_verdadero == y_predicho))


def main() -> int:
    rng = np.random.default_rng(42)
    # Datos linealmente separables
    X = np.vstack([rng.normal(-1, 1, (50, 2)), rng.normal(1, 1, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    w = ajustar(X, y, lr=1.0, epochs=2000)
    y_pred = predecir(w, X)
    print(f"w = {w}")
    print(f"Accuracy en train: {accuracy(y, y_pred):.3f}")
    print(f"Probabilidades (primeras 5): {predecir_proba(w, X[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())