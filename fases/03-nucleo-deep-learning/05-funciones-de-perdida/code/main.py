"""
Lección: 05-funciones-de-perdida
Fase: 03
Prerrequisitos: 02-redes-multicapa
"""
from __future__ import annotations
import sys
import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error. Para regresion."""
    return float(np.mean((y_true - y_pred) ** 2))


def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def mae(y_true, y_pred):
    """Mean Absolute Error. Robusto a outliers."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_derivada(y_true, y_pred):
    return np.sign(y_pred - y_true) / len(y_true)


def bce(y_true, y_pred, eps=1e-9):
    """Binary Cross-Entropy. Para clasificacion binaria.
    y_pred en (0, 1), se recorta para estabilidad."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def bce_derivada(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)


def cce(y_true, y_pred, eps=1e-9):
    """Categorical Cross-Entropy. Para multiclase.
    y_true: one-hot. y_pred: probabilidades (suma 1)."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))


def cce_derivada(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true / y_pred / y_true.shape[0]


def huber(y_true, y_pred, delta=1.0):
    """Combina MSE y MAE. Cuadratica para errores chicos, lineal para grandes."""
    error = y_true - y_pred
    abs_err = np.abs(error)
    cuad = 0.5 * error ** 2
    lin = delta * abs_err - 0.5 * delta ** 2
    return float(np.mean(np.where(abs_err <= delta, cuad, lin)))


def contraste(logits_pos, logits_neg, margen=0.0):
    """Contrastive loss (pairwise margin). logits_pos debe ser > logits_neg + margen."""
    diff = logits_pos - logits_neg
    return float(np.mean(np.maximum(0, margen - diff)))


def main() -> int:
    yt = np.array([1.0, 0.0, 1.0, 0.0])
    yp = np.array([0.9, 0.1, 0.4, 0.6])
    print(f"MSE: {mse(yt, yp):.3f}")
    print(f"MAE: {mae(yt, yp):.3f}")
    print(f"BCE: {bce(yt, yp):.3f}")
    # Multiclase
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    print(f"CCE: {cce(y_true, y_pred):.3f}")
    print(f"Huber: {huber(yt, yp):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())