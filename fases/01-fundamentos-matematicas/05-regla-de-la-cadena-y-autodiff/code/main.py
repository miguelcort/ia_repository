"""
Lección: 05-regla-de-la-cadena-y-autodiff
Fase: 01
Prerrequisitos: 04-calculo-para-ml
Fuentes:
- autograd: https://github.com/hips/autograd
- PyTorch autograd: https://pytorch.org/docs/stable/autograd.html
"""
from __future__ import annotations

import sys
from typing import Callable

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def derivada_cadena(composicion: list[Callable], x: float) -> float:
    """Aplica la regla de la cadena: si f = f_n o ... o f_1,
    entonces f'(x) = f_n'(f_{n-1}(...f_1(x))) * ... * f_1'(x)."""
    if not composicion:
        return 1.0
    valor = x
    derivadas = []
    for f in composicion:
        # Aproximamos la derivada numericamente
        h = 1e-7
        derivadas.append((f(valor + h) - f(valor - h)) / (2 * h))
        valor = f(valor)
    resultado = 1.0
    for d in derivadas:
        resultado *= d
    return resultado


def grad_check(f, x_inicial: list[float], h: float = 1e-5) -> list[float]:
    """Gradiente numerico de f: R^n -> R."""
    grad = []
    for i in range(len(x_inicial)):
        pm = list(x_inicial); pm[i] += h
        pn = list(x_inicial); pn[i] -= h
        grad.append((f(*pm) - f(*pn)) / (2 * h))
    return grad


def forward_backward_demo():
    """Mini autodiff manual: calcula gradiente de f(x) = (x-3)^2."""
    if not HAS_TORCH:
        print("PyTorch no disponible; salta demo")
        return
    x = torch.tensor([5.0], requires_grad=True)
    y = (x - 3) ** 2
    y.backward()
    print(f"  x = {x.item()}, f(x) = {y.item()}, df/dx = {x.grad.item()}")
    # df/dx = 2*(x-3) = 2*2 = 4


def main() -> int:
    f1 = lambda x: x ** 2
    f2 = lambda x: 2 * x + 1
    # composicion: x -> x^2 -> 2*(x^2) + 1
    # f(x) = 2x^2 + 1
    # f'(x) = 4x
    print("Composicion f(x) = 2*x^2 + 1")
    print(f"  f(3) = {f2(f1(3.0))}")
    print(f"  f'(3) exacta = {4 * 3}")
    print(f"  f'(3) via regla de la cadena = {derivada_cadena([f1, f2], 3.0)}")
    print()
    print("Mini-autodiff con PyTorch:")
    forward_backward_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
