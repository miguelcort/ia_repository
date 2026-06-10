"""
Lección: 02-redes-multicapa
Fase: 03
Prerrequisitos: 01-el-perceptron
"""
from __future__ import annotations
import sys
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivada(z):
    return 1 - np.tanh(z) ** 2


class CapaDensa:
    """Capa fully-connected: salida = activacion(W @ x + b)."""
    def __init__(self, n_in, n_out, activacion="sigmoid", semilla=0):
        rng = np.random.default_rng(semilla)
        # He / Xavier: escala segun activacion
        if activacion == "relu":
            escala = np.sqrt(2.0 / n_in)
        elif activacion == "tanh":
            escala = np.sqrt(1.0 / n_in)
        else:  # sigmoid
            escala = np.sqrt(1.0 / n_in)
        self.W = rng.normal(scale=escala, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activacion = activacion
        self.cache_x = None
        self.cache_z = None

    def forward(self, x):
        self.cache_x = x
        z = x @ self.W + self.b
        self.cache_z = z
        if self.activacion == "sigmoid":
            return sigmoid(z)
        elif self.activacion == "tanh":
            return tanh(z)
        elif self.activacion == "relu":
            return np.maximum(0, z)
        return z

    def backward(self, grad_salida):
        """Calcula gradiente respecto a W, b y x."""
        z = self.cache_z
        if self.activacion == "sigmoid":
            grad_act = grad_salida * sigmoid_derivada(z)
        elif self.activacion == "tanh":
            grad_act = grad_salida * tanh_derivada(z)
        elif self.activacion == "relu":
            grad_act = grad_salida * (z > 0)
        else:
            grad_act = grad_salida
        self.grad_W = self.cache_x.T @ grad_act
        self.grad_b = grad_act.sum(axis=0)
        grad_x = grad_act @ self.W.T
        return grad_x


class RedMulticapa:
    """Red feedforward multicapa."""
    def __init__(self, dims, activacion="sigmoid", semilla=0):
        # dims = [n_in, n_h1, n_h2, ..., n_out]
        self.capas = []
        for i in range(len(dims) - 1):
            self.capas.append(CapaDensa(dims[i], dims[i + 1], activacion=activacion, semilla=semilla + i))

    def forward(self, X):
        salida = X
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida

    def backward(self, grad_salida):
        grad = grad_salida
        for capa in reversed(self.capas):
            grad = capa.backward(grad)
        return grad

    def parametros(self):
        params = []
        for i, c in enumerate(self.capas):
            params.append((f"W{i}", c.W, c.grad_W))
            params.append((f"b{i}", c.b, c.grad_b))
        return params


def main() -> int:
    # XOR: red 2 -> 4 -> 1
    rng = np.random.default_rng(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    red = RedMulticapa([2, 4, 1], activacion="sigmoid", semilla=0)
    # Solo forward (sin entrenar)
    out = red.forward(X)
    print(f"Salida XOR (sin entrenar): {out.flatten()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())