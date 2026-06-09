"""
Lección: 08-optimizacion-familia-gradiente
Fase: 01
Prerrequisitos: 04-calculo-para-ml, 05-regla-de-la-cadena-y-autodiff
"""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np


def gd(f: Callable, w_inicial: list[float], lr: float = 0.1,
      max_iter: int = 1000, tol: float = 1e-6) -> tuple[list[float], list[float]]:
    """Descenso por gradiente. Devuelve (parametros_finales, historico_perdida)."""
    w = list(w_inicial)
    historial = []
    for i in range(max_iter):
        loss = f(*w)
        historial.append(loss)
        if len(historial) > 1 and abs(historial[-2] - loss) < tol:
            break
        grad = []
        h = 1e-5
        for j in range(len(w)):
            wp = list(w); wp[j] += h
            wn = list(w); wn[j] -= h
            grad.append((f(*wp) - f(*wn)) / (2 * h))
        w = [wi - lr * g for wi, g in zip(w, grad)]
    return w, historial


def gd_momentum(f, w_inicial, lr=0.1, beta=0.9, max_iter=1000, tol=1e-6):
    """GD con momentum (Polyak)."""
    w = list(w_inicial)
    v = [0.0] * len(w)
    historial = []
    for i in range(max_iter):
        loss = f(*w)
        historial.append(loss)
        if len(historial) > 1 and abs(historial[-2] - loss) < tol:
            break
        h = 1e-5
        grad = []
        for j in range(len(w)):
            wp = list(w); wp[j] += h
            wn = list(w); wn[j] -= h
            grad.append((f(*wp) - f(*wn)) / (2 * h))
        v = [beta * vi + g for vi, g in zip(v, grad)]
        w = [wi - lr * vi for wi, vi in zip(w, v)]
    return w, historial


def adam(f, w_inicial, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
         max_iter=1000, tol=1e-6):
    """Optimizador Adam: momenta de primer y segundo orden."""
    w = list(w_inicial)
    m = [0.0] * len(w)
    v = [0.0] * len(w)
    historial = []
    for t in range(1, max_iter + 1):
        loss = f(*w)
        historial.append(loss)
        if len(historial) > 1 and abs(historial[-2] - loss) < tol:
            break
        h = 1e-5
        grad = []
        for j in range(len(w)):
            wp = list(w); wp[j] += h
            wn = list(w); wn[j] -= h
            grad.append((f(*wp) - f(*wn)) / (2 * h))
        m = [beta1 * mi + (1 - beta1) * g for mi, g in zip(m, grad)]
        v = [beta2 * vi + (1 - beta2) * g ** 2 for vi, g in zip(v, grad)]
        m_hat = [mi / (1 - beta1 ** t) for mi in m]
        v_hat = [vi / (1 - beta2 ** t) for vi in v]
        w = [wi - lr * mh / (vh ** 0.5 + eps) for wi, mh, vh in zip(w, m_hat, v_hat)]
    return w, historial


def main() -> int:
    # Funcion cuadratica simple
    f = lambda x, y: (x - 3) ** 2 + (y + 2) ** 2
    print("Minimo de f(x,y) = (x-3)^2 + (y+2)^2, esperado en [3, -2]")
    for nombre, opt in [("GD", gd), ("GD+Momentum", gd_momentum), ("Adam", adam)]:
        w, hist = opt(f, [0.0, 0.0], max_iter=200)
        print(f"  {nombre:15s} -> w={[round(x, 4) for x in w]}, "
              f"perdida_final={hist[-1]:.2e}, iter={len(hist)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
