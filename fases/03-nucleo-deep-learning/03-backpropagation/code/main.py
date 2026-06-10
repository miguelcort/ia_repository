"""
Lección: 03-backpropagation
Fase: 03
Prerrequisitos: 02-redes-multicapa
"""
from __future__ import annotations
import sys
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z):
    s = sigmoid(z)
    return s * (1 - s)


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


class RedBP:
    """Red feedforward con backprop completo y optimizador SGD."""
    def __init__(self, dims, lr=0.1, semilla=0):
        rng = np.random.default_rng(semilla)
        self.lr = lr
        self.capas = []
        for i in range(len(dims) - 1):
            escala = np.sqrt(1.0 / dims[i])
            W = rng.normal(scale=escala, size=(dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.capas.append({"W": W, "b": b, "x": None, "z": None, "a": None})

    def forward(self, X):
        entrada = X
        for i, capa in enumerate(self.capas):
            capa["x"] = entrada
            capa["z"] = entrada @ capa["W"] + capa["b"]
            capa["a"] = sigmoid(capa["z"])
            entrada = capa["a"]
        return entrada

    def backward(self, y_true, y_pred):
        n = len(y_true)
        grad = 2 * (y_pred - y_true) / n * sigmoid_derivada(self.capas[-1]["z"])
        for i in reversed(range(len(self.capas))):
            capa = self.capas[i]
            capa["grad_W"] = capa["x"].T @ grad
            capa["grad_b"] = grad.sum(axis=0)
            if i > 0:
                grad = grad @ capa["W"].T * sigmoid_derivada(self.capas[i - 1]["z"])
        return self

    def step(self):
        for capa in self.capas:
            capa["W"] -= self.lr * capa["grad_W"]
            capa["b"] -= self.lr * capa["grad_b"]

    def fit(self, X, y, epocas=1000, verbose=True):
        for epoca in range(epocas):
            y_pred = self.forward(X)
            self.backward(y, y_pred)
            self.step()
            if verbose and epoca % 200 == 0:
                print(f"Epoca {epoca}: mse={mse(y, y_pred):.4f}")
        return self


def main() -> int:
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    red = RedBP([2, 4, 1], lr=1.0)
    red.fit(X, y, epocas=2000)
    print("Predicciones XOR:")
    print(red.forward(X).round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())