"""
Lección: 08-inicializacion-de-pesos
Fase: 03
Prerrequisitos: 02-redes-multicapa
"""
from __future__ import annotations
import sys
import numpy as np


def ceros(n_in, n_out):
    """Inicializacion a cero. NO usar: gradiente cero, red no aprende."""
    return np.zeros((n_in, n_out))


def aleatorio_simple(n_in, n_out, semilla=0):
    """Inicializacion aleatoria uniforme [0, 1]. Suboptima."""
    rng = np.random.default_rng(semilla)
    return rng.uniform(0, 1, size=(n_in, n_out))


def xavier_glorot(n_in, n_out, semilla=0):
    """Xavier/Glorot: var = 1 / n_in. Para sigmoid/tanh.
    Mantiene varianza estable forward+backward."""
    rng = np.random.default_rng(semilla)
    escala = np.sqrt(1.0 / n_in)
    return rng.normal(scale=escala, size=(n_in, n_out))


def he(n_in, n_out, semilla=0):
    """He: var = 2 / n_in. Para ReLU y variantes.
    Corrige la varianza considerando que ReLU 'mata' la mitad de la senal."""
    rng = np.random.default_rng(semilla)
    escala = np.sqrt(2.0 / n_in)
    return rng.normal(scale=escala, size=(n_in, n_out))


def lecun(n_in, n_out, semilla=0):
    """LeCun: var = 1 / n_in. Para SELU y tanh."""
    rng = np.random.default_rng(semilla)
    escala = np.sqrt(1.0 / n_in)
    return rng.normal(scale=escala, size=(n_in, n_out))


def varianza_activaciones(X_in, W, n_pasos=10):
    """Simula varianza de activaciones a traves de N capas con ReLU.
    Mide estabilidad: si crece o cae, hay problema de inicializacion."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, W.shape[0]))
    vars_ = []
    for _ in range(n_pasos):
        z = x @ W
        x = np.maximum(0, z)
        vars_.append(float(x.var()))
    return vars_


def main() -> int:
    n_in, n_out = 100, 100
    for nombre, fn in [("Xavier", xavier_glorot), ("He", he), ("LeCun", lecun), ("Ceros", ceros)]:
        W = fn(n_in, n_out)
        vars_ = varianza_activaciones(n_in, W, n_pasos=10)
        print(f"{nombre}: var_inicial={vars_[0]:.3f}, var_final={vars_[-1]:.3f}, max={max(vars_):.2f}, min={min(vars_):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())