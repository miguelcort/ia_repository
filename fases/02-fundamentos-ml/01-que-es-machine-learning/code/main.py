"""
Lección: 01-que-es-machine-learning
Fase: 02
Prerrequisitos: 00-configuracion-y-herramientas, 01-fundamentos-matematicas
"""
from __future__ import annotations
import sys
import numpy as np


def generar_datos_lineales(n=100, ruido=0.1, semilla=0):
    """Genera y = 2x + 1 + ruido."""
    rng = np.random.default_rng(semilla)
    X = np.linspace(0, 5, n)
    y = 2 * X + 1 + ruido * rng.normal(size=n)
    return X, y


def error_cuadratico_medio(y_verdadero, y_predicho):
    return float(np.mean((y_verdadero - y_predicho) ** 2))


def dividir_train_test(X, y, prop_train=0.8, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(X)
    indices = rng.permutation(n)
    n_train = int(n * prop_train)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def main() -> int:
    X, y = generar_datos_lineales()
    print(f"X.shape={X.shape}, y.shape={y.shape}")
    X_tr, y_tr, X_te, y_te = dividir_train_test(X, y)
    print(f"Train: {len(X_tr)} muestras, Test: {len(X_te)} muestras")
    y_pred = 2 * X_te + 1  # Prediccion perfecta (sin ruido)
    print(f"ECM en test (perfecto): {error_cuadratico_medio(y_te, y_pred):.6f}")
    y_naive = np.full_like(y_te, y.mean())
    print(f"ECM en test (predict media): {error_cuadratico_medio(y_te, y_naive):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())