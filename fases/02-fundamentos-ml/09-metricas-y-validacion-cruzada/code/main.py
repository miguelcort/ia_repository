"""
Lección: 09-metricas-y-validacion-cruzada
Fase: 02
Prerrequisitos: 02-regresion-lineal-desde-cero, 03-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


def accuracy(y_v, y_p):
    return float(np.mean(y_v == y_p))


def precision(y_v, y_p, clase=1):
    tp = np.sum((y_v == clase) & (y_p == clase))
    fp = np.sum((y_v != clase) & (y_p == clase))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_v, y_p, clase=1):
    tp = np.sum((y_v == clase) & (y_p == clase))
    fn = np.sum((y_v == clase) & (y_p != clase))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1(y_v, y_p, clase=1):
    p = precision(y_v, y_p, clase)
    r = recall(y_v, y_p, clase)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def k_fold(n, k, semilla=0):
    """Genera indices para K-fold CV."""
    rng = np.random.default_rng(semilla)
    indices = rng.permutation(n)
    return np.array_split(indices, k)


def cross_val_score(modelo_fn, X, y, k=5, semilla=0):
    """Evalua modelo_fn con K-fold CV. modelo_fn(X_tr, y_tr, X_te) -> y_pred."""
    scores = []
    for fold in k_fold(len(X), k, semilla):
        mask = np.ones(len(X), dtype=bool)
        mask[fold] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[fold], y[fold]
        y_pred = modelo_fn(X_tr, y_tr, X_te)
        scores.append(accuracy(y_te, y_pred))
    return np.array(scores)


def main() -> int:
    rng = np.random.default_rng(0)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_p = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    print(f"Accuracy: {accuracy(y, y_p):.3f}")
    print(f"Precision: {precision(y, y_p):.3f}")
    print(f"Recall: {recall(y, y_p):.3f}")
    print(f"F1: {f1(y, y_p):.3f}")
    scores = cross_val_score(lambda Xt, yt, Xe: np.zeros(len(Xe), dtype=int), rng.normal(size=(50, 2)), rng.integers(0, 2, 50))
    print(f"5-fold CV (modelo dummy): {scores.mean():.3f} +/- {scores.std():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())