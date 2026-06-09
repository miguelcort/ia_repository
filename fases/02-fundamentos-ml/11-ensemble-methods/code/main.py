"""
Lección: 11-ensemble-methods
Fase: 02
Prerrequisitos:** 04-arboles-de-decision-random-forest
"""
from __future__ import annotations
import sys
import numpy as np


def voting_clasificacion(predicciones_list):
    """Voto mayoritario: cada modelo vota, gana la clase mas votada."""
    predicciones = np.array(predicciones_list)
    valores, counts = np.unique(predicciones, return_counts=True)
    return valores[np.argmax(counts)]


def bagging(estimador_fn, X, y, n_estimadores=10, muestra_frac=0.8, semilla=0):
    """Bagging: cada estimador ve un subconjunto bootstrap de los datos."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    estimadores = []
    for _ in range(n_estimadores):
        idx = rng.choice(n, size=int(n * muestra_frac), replace=True)
        estimador = estimador_fn(X[idx], y[idx])
        estimadores.append(estimador)
    return estimadores


def boosting_pesos(estimador_fn, X, y, n_rondas=10):
    """Boosting simple: cada muestra tiene un peso; los errores suben de peso."""
    n = len(X)
    pesos = np.ones(n) / n
    estimadores = []
    alpha = []
    for _ in range(n_rondas):
        idx = np.random.choice(n, size=n, replace=True, p=pesos)
        estimador = estimador_fn(X[idx], y[idx])
        pred = estimador(X)
        error = np.sum(pesos * (pred != y)) / pesos.sum()
        if error >= 0.5 or error == 0:
            break
        a = 0.5 * np.log((1 - error) / error)
        pesos = pesos * np.exp(a * (pred != y))
        pesos /= pesos.sum()
        estimadores.append(estimador)
        alpha.append(a)
    return estimadores, alpha


def predecir_boosting(estimadores, alpha, X):
    """Prediccion = signo(sum alpha_i * h_i(x))."""
    n = len(X)
    votos = np.zeros(n)
    for est, a in zip(estimadores, alpha):
        votos = votos + a * (2 * est(X) - 1)
    return (votos > 0).astype(int)


def main() -> int:
    # 3 modelos, voto
    y1 = np.array([0, 1, 0, 1, 0])
    y2 = np.array([0, 0, 1, 1, 0])
    y3 = np.array([0, 1, 0, 1, 1])
    for x in range(5):
        voto = voting_clasificacion([y1[x], y2[x], y3[x]])
        print(f"x={x}: votos={[y1[x], y2[x], y3[x]]} -> {voto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())