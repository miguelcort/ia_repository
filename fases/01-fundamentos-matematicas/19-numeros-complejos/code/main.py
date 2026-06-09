"""
Lección: 19-numeros-complejos
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
"""
from __future__ import annotations
import sys
import cmath
import math


def sumar(a, b):
    return (a[0] + b[0], a[1] + b[1])


def multiplicar(a, b):
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    real = a[0] * b[0] - a[1] * b[1]
    imag = a[0] * b[1] + a[1] * b[0]
    return (real, imag)


def modulo(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)


def argumento(a):
    return math.atan2(a[1], a[0])


def polar_a_rect(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))


def fft_magnitud(senal):
    """Magnitud del espectro de Fourier."""
    import numpy as np
    return np.abs(np.fft.rfft(senal))


def main() -> int:
    a = (1.0, 2.0)
    b = (3.0, 4.0)
    print(f"a + b = {sumar(a, b)}")
    print(f"a * b = {multiplicar(a, b)}")
    print(f"|a| = {modulo(a):.4f}")
    print(f"arg(a) = {argumento(a):.4f}")
    print(f"polar(2, pi/4) = {polar_a_rect(2, math.pi / 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())