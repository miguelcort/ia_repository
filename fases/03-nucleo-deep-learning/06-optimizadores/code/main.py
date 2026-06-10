"""
Lección: 06-optimizadores
Fase: 03
Prerrequisitos: 03-backpropagation
"""
from __future__ import annotations
import sys
import numpy as np


class SGD:
    """Stochastic Gradient Descent con learning rate."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def actualizar(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class SGD_Momentum:
    """SGD con momentum. Acelera convergencia y suaviza oscilaciones."""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vel = None

    def actualizar(self, params, grads):
        if self.vel is None:
            self.vel = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.vel[i] = self.momentum * self.vel[i] + g
            p -= self.lr * self.vel[i]


class AdaGrad:
    """Adagrad. Adapta lr por parametro segun suma de gradientes^2."""
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.sum_sq = None

    def actualizar(self, params, grads):
        if self.sum_sq is None:
            self.sum_sq = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.sum_sq[i] += g ** 2
            p -= self.lr * g / (np.sqrt(self.sum_sq[i]) + self.eps)


class RMSProp:
    """RMSProp. Media movil exponencial de gradientes^2."""
    def __init__(self, lr=0.001, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.media = None

    def actualizar(self, params, grads):
        if self.media is None:
            self.media = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.media[i] = self.decay * self.media[i] + (1 - self.decay) * g ** 2
            p -= self.lr * g / (np.sqrt(self.media[i]) + self.eps)


class Adam:
    """Adam: momentum + RMSProp + bias correction. Estándar en deep learning."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # primer momento
        self.v = None  # segundo momento
        self.t = 0

    def actualizar(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def optimizar(optim, f, df, x0, n_iter=100):
    """Minimiza f(x) con derivada df. Devuelve la trayectoria."""
    x = x0.copy()
    trayectoria = [x.copy()]
    for _ in range(n_iter):
        grads = [df(x)]
        optim.actualizar([x], grads)
        trayectoria.append(x.copy())
    return x, np.array(trayectoria)


def main() -> int:
    # Minimizar f(x) = x^2. df = 2x.
    f = lambda x: x ** 2
    df = lambda x: 2 * x
    for nombre, opt in [("SGD", SGD(0.1)),
                        ("Momentum", SGD_Momentum(0.1, 0.9)),
                        ("Adam", Adam(0.1))]:
        x, _ = optimizar(opt, f, df, np.array([10.0]), n_iter=50)
        print(f"{nombre}: x final = {x[0]:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())