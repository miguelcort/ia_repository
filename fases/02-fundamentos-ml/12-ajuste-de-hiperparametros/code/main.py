"""
Lección: 12-ajuste-de-hiperparametros
Fase: 02
Prerrequisitos: 05-evaluacion-de-modelos-y-overfitting
"""
from __future__ import annotations
import sys
import numpy as np


def k_folds(n, k=5, semilla=0):
    """Genera k folds deterministicos."""
    rng = np.random.default_rng(semilla)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def cross_val_score(modelo_fn, X, y, k=5, semilla=0):
    """Evalua un modelo con k-fold CV. Devuelve scores por fold."""
    folds = k_folds(len(X), k, semilla)
    scores = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        y_pred = modelo_fn(X[train], y[train], X[val])
        if y[val].dtype == int or (y[val] == y[val].astype(int)).all():
            score = float(np.mean(y_pred == y[val]))
        else:
            score = 1 - float(np.mean((y[val] - y_pred) ** 2))
        scores.append(score)
    return np.array(scores)


def grid_search(modelo_fn_factory, X, y, params_grid, k=5, semilla=0):
    """Grid search: evalua todas las combinaciones de hiperparametros."""
    keys = list(params_grid.keys())
    valores = [params_grid[k_] for k_ in keys]
    resultados = []
    for combo in _producto(valores):
        kwargs = dict(zip(keys, combo))
        modelo = modelo_fn_factory(**kwargs)
        scores = cross_val_score(modelo, X, y, k, semilla)
        resultados.append({"params": kwargs, "media": float(scores.mean()), "std": float(scores.std())})
    return resultados


def _producto(listas):
    """Producto cartesiano de una lista de listas."""
    if not listas:
        yield ()
        return
    primero, resto = listas[0], listas[1:]
    for r in _producto(resto):
        for p in primero:
            yield (p,) + r


def random_search(modelo_fn_factory, X, y, distribuciones, n_iter=10, k=5, semilla=0):
    """Random search: muestrea N combinaciones aleatorias."""
    rng = np.random.default_rng(semilla)
    keys = list(distribuciones.keys())
    resultados = []
    for _ in range(n_iter):
        kwargs = {k_: distribuciones[k_](rng) for k_ in keys}
        modelo = modelo_fn_factory(**kwargs)
        scores = cross_val_score(modelo, X, y, k, semilla)
        resultados.append({"params": kwargs, "media": float(scores.mean()), "std": float(scores.std())})
    return resultados


def mejor(resultados):
    """Devuelve la combinacion con mejor media."""
    return max(resultados, key=lambda r: r["media"])


def main() -> int:
    rng = np.random.default_rng(0)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.squeeze() + 1 + 0.5 * rng.normal(size=50)
    # Modelo lineal con hiperparametro: tipo de regularizacion (mock)
    def modelo_factory(grado=1):
        def m(Xt, yt, Xe):
            coef = np.polyfit(Xt.squeeze(), yt, grado)
            return np.polyval(coef, Xe.squeeze())
        return m
    grid = {"grado": [1, 2, 3, 5, 8]}
    res = grid_search(modelo_factory, X, y, grid, k=5)
    print("Grid search:")
    for r in res:
        print(f"  {r['params']}: media={r['media']:.4f} std={r['std']:.4f}")
    print(f"Mejor: {mejor(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())