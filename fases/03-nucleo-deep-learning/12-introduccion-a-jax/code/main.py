"""
Lección: 12-introduccion-a-jax
Fase: 03
Prerrequisitos: 11-introduccion-a-pytorch
"""
from __future__ import annotations
import sys
import numpy as np


def main() -> int:
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, vmap
        disponible = True
    except ImportError:
        disponible = False
    if not disponible:
        print("JAX no esta instalado. Esta leccion requiere:")
        print("  pip install jax jaxlib")
        print("Conceptos cubiertos (sin codigo ejecutable):")
        print("  - jax.numpy: numpy con autodiff")
        print("  - grad(f): gradiente automatico de f")
        print("  - jit(f): compilacion XLA para performance")
        print("  - vmap(f): vectorizacion automatica")
        print("  - pmap(f): paralelismo multi-device")
        return 0

    # Ejemplo: gradiente automatico
    def f(x):
        return jnp.sum(x ** 2)

    g = grad(f)
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"f(x) = {f(x)}")
    print(f"grad(f)(x) = {g(x)}")

    # JIT
    f_jit = jit(f)
    print(f"jit(f)(x) = {f_jit(x)}")

    # vmap
    matriz = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    f_vmapped = vmap(f)
    print(f"vmap(f)(matriz) = {f_vmapped(matriz)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())