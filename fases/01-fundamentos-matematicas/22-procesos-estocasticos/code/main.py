"""
Lección: 22-procesos-estocasticos
Fase: 01
Prerrequisitos: 06-probabilidad-y-distribuciones
"""
from __future__ import annotations
import sys
import numpy as np


def caminata_aleatoria(n_pasos, semilla=0):
    """Caminata aleatoria 1D: x_0 = 0, x_{t+1} = x_t + N(0, 1)."""
    rng = np.random.default_rng(semilla)
    incrementos = rng.normal(0, 1, n_pasos)
    return np.cumsum(incrementos)


def cadena_markov(estados, transiciones, n_pasos, inicio, semilla=0):
    """Cadena de Markov: P(s_{t+1} | s_t) dada por matriz de transicion."""
    rng = np.random.default_rng(semilla)
    actual = inicio
    historia = [actual]
    for _ in range(n_pasos - 1):
        actual = rng.choice(estados, p=transiciones[actual])
        historia.append(actual)
    return historia


def media_varianza_caminata(n_pasos, n_simulaciones, semilla=0):
    rng = np.random.default_rng(semilla)
    finales = []
    for i in range(n_simulaciones):
        s = rng.normal(0, 1, n_pasos)
        finales.append(s.sum())
    return float(np.mean(finales)), float(np.var(finales))


def main() -> int:
    trayectoria = caminata_aleatoria(1000, semilla=42)
    print(f"Cam. aleatoria 1000 pasos: posicion final = {trayectoria[-1]:.3f}")
    mu, var = media_varianza_caminata(100, 10000, semilla=0)
    print(f"Distribucion final: media={mu:.4f} (esperado ~0), var={var:.4f} (esperado ~100)")
    estados = ["A", "B", "C"]
    trans = {
        "A": [0.1, 0.6, 0.3],
        "B": [0.4, 0.4, 0.2],
        "C": [0.3, 0.3, 0.4],
    }
    historia = cadena_markov(estados, trans, 100, "A", semilla=0)
    print(f"Markov: ultimos 5 = {historia[-5:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())