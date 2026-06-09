"""
Lección: 17-datos-desbalanceados
Fase: 02
Prerrequisitos:** 09-metricas-y-validacion-cruzada
"""
from __future__ import annotations
import sys
import numpy as np


def oversampling_simple(X, y, ratio=1.0, semilla=0):
    """Sobremuestreo aleatorio: replica muestras de la clase minoritaria.
    No sintetiza datos nuevos.
    """
    rng = np.random.default_rng(semilla)
    clases, counts = np.unique(y, return_counts=True)
    mayor = clases[counts.argmax()]
    menor = clases[counts.argmin()]
    n_mayor = counts.max()
    n_menor = counts.min()
    n_objetivo = int(n_mayor * ratio)
    n_replicas = max(0, n_objetivo - n_menor)
    idx_menor = np.where(y == menor)[0]
    if n_replicas > 0 and len(idx_menor) > 0:
        idx_nuevos = rng.choice(idx_menor, size=n_replicas, replace=True)
        X_new = np.vstack([X, X[idx_nuevos]])
        y_new = np.concatenate([y, y[idx_nuevos]])
    else:
        X_new, y_new = X, y
    return X_new, y_new


def undersampling_simple(X, y, ratio=1.0, semilla=0):
    """Submuestreo aleatorio: reduce la clase mayoritaria."""
    rng = np.random.default_rng(semilla)
    clases, counts = np.unique(y, return_counts=True)
    mayor = clases[counts.argmax()]
    menor = clases[counts.argmin()]
    n_menor = counts.min()
    n_objetivo = int(n_menor * ratio)
    idx_mayor = np.where(y == mayor)[0]
    idx_menor = np.where(y == menor)[0]
    if n_objetivo < len(idx_mayor):
        idx_muestra = rng.choice(idx_mayor, size=n_objetivo, replace=False)
        X_new = np.vstack([X[idx_menor], X[idx_muestra]])
        y_new = np.concatenate([y[idx_menor], y[idx_muestra]])
    else:
        X_new, y_new = X, y
    return X_new, y_new


def pesos_por_clase(y):
    """Devuelve w_c = N / (n_clases * n_c) para usar en sample_weight."""
    clases, counts = np.unique(y, return_counts=True)
    n = len(y)
    n_clases = len(clases)
    pesos = np.zeros(n)
    for c, count in zip(clases, counts):
        w = n / (n_clases * count)
        pesos[y == c] = w
    return pesos


def confusion_matriz(y_true, y_pred, positiva=1):
    """Calcula TP, FP, FN, TN para clase positiva binaria."""
    tp = int(np.sum((y_true == positiva) & (y_pred == positiva)))
    fp = int(np.sum((y_true != positiva) & (y_pred == positiva)))
    fn = int(np.sum((y_true == positiva) & (y_pred != positiva)))
    tn = int(np.sum((y_true != positiva) & (y_pred != positiva)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def f1_score(cm):
    """F1 a partir de una confusion matriz binaria."""
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    # Datos desbalanceados: 100 clase 0, 10 clase 1
    rng = np.random.default_rng(0)
    X0 = rng.normal(size=(100, 2))
    X1 = rng.normal(loc=[2, 2], size=(10, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 10)
    # Oversample
    Xo, yo = oversampling_simple(X, y, ratio=1.0)
    print(f"Oversample: {dict(zip(*np.unique(yo, return_counts=True)))}")
    # Undersample
    Xu, yu = undersampling_simple(X, y, ratio=1.0)
    print(f"Undersample: {dict(zip(*np.unique(yu, return_counts=True)))}")
    # Pesos
    pesos = pesos_por_clase(y)
    print(f"Clase 0 peso medio: {pesos[y==0].mean():.3f}, clase 1: {pesos[y==1].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())