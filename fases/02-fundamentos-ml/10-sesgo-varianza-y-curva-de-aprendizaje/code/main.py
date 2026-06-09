"""
Lección: 10-sesgo-varianza-y-curva-de-aprendizaje
Fase: 02
Prerrequisitos: 09-metricas-y-validacion-cruzada
"""
from __future__ import annotations
import sys
import numpy as np


def mse(y_v, y_p):
    return float(np.mean((y_v - y_p) ** 2))


def sesgo_varianza(estimador_fn, X, y, n_boot=50, semilla=0):
    """Estima sesgo^2 y varianza de un estimador por bootstrap."""
    rng = np.random.default_rng(semilla)
    preds = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        pred = estimador_fn(X[idx], y[idx], X)
        preds.append(pred)
    preds = np.array(preds)
    media_pred = preds.mean(axis=0)
    sesgo = float(np.mean((media_pred - y) ** 2))
    varianza = float(np.mean(preds.var(axis=0)))
    return sesgo, varianza


def curva_aprendizaje(modelo_fn, X, y, k=5):
    """Train scores vs tamano del train set."""
    n = len(X)
    sizes = np.linspace(10, n, k, dtype=int)
    train_scores, val_scores = [], []
    for s in sizes:
        Xtr, ytr = X[:s], y[:s]
        Xte, yte = X[s:], y[s:]
        if len(Xte) == 0:
            break
        y_pred = modelo_fn(Xtr, ytr, Xte)
        val_scores.append(1 - mse(yte, y_pred))
        y_pred_tr = modelo_fn(Xtr, ytr, Xtr)
        train_scores.append(1 - mse(ytr, y_pred_tr))
    return sizes[:len(train_scores)], np.array(train_scores), np.array(val_scores)


def main() -> int:
    rng = np.random.default_rng(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.squeeze() + 1 + 0.5 * rng.normal(size=50)
    sizes, tr, va = curva_aprendizaje(lambda Xt, yt, Xe: 2 * Xe.squeeze() + 1, X, y, k=5)
    print(f"Tamanos: {sizes}")
    print(f"Train R^2: {tr}")
    print(f"Val R^2: {va}")
    return 0


if __name__ == "__main__":
    sys.exit(main())