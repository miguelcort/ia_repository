"""
Lección: 18-optimizacion-convexa
Fase: 01
Prerrequisitos: 08-optimizacion-familia-gradiente
"""
from __future__ import annotations
import sys
import numpy as np


def es_convexa_2d(f, x, h=1e-5):
    """Verifica convexidad aproximada via Hessiana 2x2."""
    fxx = (f(x[0] + h, x[1]) - 2 * f(x[0], x[1]) + f(x[0] - h, x[1])) / h**2
    fyy = (f(x[0], x[1] + h) - 2 * f(x[0], x[1]) + f(x[0], x[1] - h)) / h**2
    fxy = (f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h) - f(x[0] - h, x[1] + h) + f(x[0] - h, x[1] - h)) / (4 * h**2)
    hess = np.array([[fxx, fxy], [fxy, fyy]])
    autovalores = np.linalg.eigvalsh(hess)
    return bool(np.all(autovalores >= -1e-6)), autovalores.tolist()


def gradiente_descenso_convexo(f, w_inicial, lr=0.1, max_iter=100):
    """Para funciones convexas, converge al minimo global."""
    w = np.array(w_inicial, dtype=float)
    h = 1e-5
    for _ in range(max_iter):
        grad = np.zeros_like(w)
        for i in range(len(w)):
            wp = w.copy(); wp[i] += h
            wn = w.copy(); wn[i] -= h
            grad[i] = (f(*wp) - f(*wn)) / (2 * h)
        w = w - lr * grad
    return w


def main() -> int:
    # Funcion estrictamente convexa: f(x,y) = (x-1)^2 + (y+1)^2
    f = lambda x, y: (x - 1) ** 2 + (y + 1) ** 2
    conv, autovals = es_convexa_2d(f, [0.0, 0.0])
    print(f"Convexa: {conv}, autovalores: {autovals}")
    w = gradiente_descenso_convexo(f, [0.0, 0.0], lr=0.1, max_iter=200)
    print(f"Minimo encontrado: {w} (esperado [1, -1])")
    return 0


if __name__ == "__main__":
    sys.exit(main())