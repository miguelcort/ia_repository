"""
Lección: 10-mini-framework
Fase: 03
Prerrequisitos: 09-programacion-de-learning-rate
"""
from __future__ import annotations
import sys
import numpy as np


class Modulo:
    """Clase base. Cada modulo implementa forward y backward."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def backward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Modulo):
    """Capa densa: y = x @ W + b."""
    def __init__(self, n_in, n_out, semilla=0):
        rng = np.random.default_rng(semilla)
        escala = np.sqrt(2.0 / n_in)  # He
        self.W = rng.normal(scale=escala, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.cache_x = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.cache_x = x
        return x @ self.W + self.b

    def backward(self, grad_salida):
        self.grad_W = self.cache_x.T @ grad_salida
        self.grad_b = grad_salida.sum(axis=0)
        return grad_salida @ self.W.T


class ReLU(Modulo):
    def forward(self, x):
        self.cache_x = x
        return np.maximum(0, x)

    def backward(self, grad_salida):
        return grad_salida * (self.cache_x > 0)


class Sigmoid(Modulo):
    def forward(self, x):
        self.cache_out = 1.0 / (1.0 + np.exp(-x))
        return self.cache_out

    def backward(self, grad_salida):
        return grad_salida * self.cache_out * (1 - self.cache_out)


class Sequential:
    """Encadena modulos. Aplica forward y backward en orden."""
    def __init__(self, *modulos):
        self.modulos = list(modulos)

    def forward(self, x):
        for m in self.modulos:
            x = m.forward(x)
        return x

    def backward(self, grad):
        for m in reversed(self.modulos):
            grad = m.backward(grad)
        return grad

    def parametros(self):
        """Devuelve [(nombre, param, grad), ...] para el optimizador."""
        params = []
        for i, m in enumerate(self.modulos):
            if isinstance(m, Linear):
                params.append((f"{i}.W", m.W, m.grad_W))
                params.append((f"{i}.b", m.b, m.grad_b))
        return params


class Entrenador:
    """Loop de entrenamiento completo: forward, backward, step, scheduler."""
    def __init__(self, modelo, optim, loss_fn, loss_derivada, scheduler=None):
        self.modelo = modelo
        self.optim = optim
        self.loss_fn = loss_fn
        self.loss_derivada = loss_derivada
        self.scheduler = scheduler
        self.historial = []

    def paso(self, X, y):
        y_pred = self.modelo.forward(X)
        loss = self.loss_fn(y, y_pred)
        grad = self.loss_derivada(y, y_pred)
        self.modelo.backward(grad)
        params_grads = [(p, g) for (_, p, g) in self.modelo.parametros()]
        self.optim.actualizar([p for p, _ in params_grads],
                              [g for _, g in params_grads])
        if self.scheduler is not None:
            self.scheduler()
        return loss

    def fit(self, X, y, epocas=100, verbose=True):
        for epoca in range(epocas):
            loss = self.paso(X, y)
            self.historial.append(loss)
            if verbose and epoca % 10 == 0:
                print(f"Epoca {epoca}: loss={loss:.4f}")
        return self


class AdamSimple:
    """Adam minimo para auto-suficiencia."""
    def __init__(self, lr=0.001):
        self.lr = lr
        self.m = None
        self.v = None
        self.t = 0

    def actualizar(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g ** 2)
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def mse_derivada(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def main() -> int:
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    modelo = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )
    trainer = Entrenador(modelo, AdamSimple(0.05), mse, mse_derivada)
    trainer.fit(X, y, epocas=500, verbose=False)
    pred = modelo.forward(X)
    print(f"XOR final: {pred.flatten().round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())