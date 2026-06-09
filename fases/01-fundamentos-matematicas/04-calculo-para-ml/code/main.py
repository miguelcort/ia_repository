"""
Lección: 04-calculo-para-ml
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
Fuentes:
- sympy: https://docs.sympy.org/
- "Calculus" (3Blue1Brown): https://www.3blue1brown.com/topics/calculus
"""
from __future__ import annotations

import sys


def derivada_polinomio(coefs: list[float]) -> list[float]:
    """Dado un polinomio [a0, a1, a2, ...] que representa
    a0 + a1*x + a2*x^2 + ..., devuelve los coeficientes de su
    derivada: [a1, 2*a2, 3*a3, ...]."""
    return [coefs[i] * i for i in range(1, len(coefs))]


def evaluar_polinomio(coefs: list[float], x: float) -> float:
    resultado = 0.0
    for i, c in enumerate(coefs):
        resultado += c * (x ** i)
    return resultado


def derivada_numerica(f, x: float, h: float = 1e-7) -> float:
    """Aproxima f'(x) con diferencia central."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradiente(f, vars_inicial: list[float], h: float = 1e-5, max_iter: int = 100,
              lr: float = 0.1) -> list[float]:
    """Descenso por gradiente numerico en R^n."""
    vars_act = list(vars_inicial)
    for _ in range(max_iter):
        grad = []
        for i in range(len(vars_act)):
            punto_mas = list(vars_act)
            punto_mas[i] += h
            punto_menos = list(vars_act)
            punto_menos[i] -= h
            g = (f(*punto_mas) - f(*punto_menos)) / (2 * h)
            grad.append(g)
        vars_act = [v - lr * g for v, g in zip(vars_act, grad)]
    return vars_act


def main() -> int:
    # f(x) = 3x^2 + 2x + 1 -> f'(x) = 6x + 2
    f = lambda x: 3 * x ** 2 + 2 * x + 1
    print("f(x) = 3x^2 + 2x + 1")
    print(f"  f'(2) exacta = {6 * 2 + 2}")
    print(f"  f'(2) numerica = {derivada_numerica(f, 2.0)}")
    # Minimo de f(x,y) = (x-1)^2 + (y+2)^2
    g = lambda x, y: (x - 1) ** 2 + (y + 2) ** 2
    minimo = gradiente(g, [0.0, 0.0])
    print(f"  Minimo numerico de (x-1)^2 + (y+2)^2: {minimo}")
    print(f"  (esperado cerca de [1, -2])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
