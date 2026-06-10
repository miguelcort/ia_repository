"""
Lección: 01-el-perceptron
Fase: 03
Prerrequisitos: 02-fundamentos-ml/02-modelos-lineales-y-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


class Perceptron:
    """Perceptrón de Rosenblatt. Clasificador binario lineal."""
    def __init__(self, n_features, lr=0.1, n_epocas=100, semilla=0):
        rng = np.random.default_rng(semilla)
        self.pesos = rng.normal(scale=0.01, size=n_features)
        self.sesgo = 0.0
        self.lr = lr
        self.n_epocas = n_epocas
        self.historial = []

    def activar(self, z):
        """Step function: 1 si z >= 0, sino 0."""
        return np.where(z >= 0, 1, 0)

    def predecir(self, X):
        z = X @ self.pesos + self.sesgo
        return self.activar(z)

    def fit(self, X, y):
        """Regla del perceptron: actualiza pesos cuando hay error.
        w <- w + lr * (y - y_pred) * x.
        """
        self.historial = []
        y_bin = np.where(y > 0, 1, 0)
        for _ in range(self.n_epocas):
            errores = 0
            for xi, yi in zip(X, y_bin):
                y_pred = self.predecir(xi.reshape(1, -1))[0]
                if y_pred != yi:
                    self.pesos += self.lr * (yi - y_pred) * xi
                    self.sesgo += self.lr * (yi - y_pred)
                    errores += 1
            self.historial.append(errores)
            if errores == 0:
                break
        return self

    def score(self, X, y):
        """Accuracy."""
        y_pred = self.predecir(X)
        y_bin = np.where(y > 0, 1, 0)
        return float(np.mean(y_pred == y_bin))


def main() -> int:
    # AND logico
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    p = Perceptron(n_features=2, lr=0.1, n_epocas=20)
    p.fit(X, y)
    print(f"AND accuracy: {p.score(X, y):.2f}")
    print(f"Predicciones: {p.predecir(X)}")
    print(f"Epocas con error: {p.historial}")
    return 0


if __name__ == "__main__":
    sys.exit(main())