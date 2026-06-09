"""
Lección: 06-probabilidad-y-distribuciones
Fase: 01
Prerrequisitos: 01-intuicion-algebra-lineal
Fuentes:
- NumPy random: https://numpy.org/doc/stable/reference/random/index.html
- "Information Theory, Inference, and Learning Algorithms" (MacKay)
"""
from __future__ import annotations

import sys

import numpy as np


def bernoulli(media: float) -> callable:
    """Distribución de Bernoulli: P(X=1) = media, P(X=0) = 1-media."""
    p = media
    def pmf(x):
        if x == 0:
            return 1 - p
        if x == 1:
            return p
        return 0.0
    return pmf


def binomial(n: int, p: float) -> callable:
    """Distribución binomial."""
    from math import comb
    def pmf(k):
        if k < 0 or k > n:
            return 0.0
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return pmf


def gaussiana(mu: float = 0.0, sigma: float = 1.0) -> callable:
    """Distribución normal con media mu y desv sigma."""
    from math import exp, pi, sqrt
    coef = 1 / (sigma * sqrt(2 * pi))
    def pdf(x):
        return coef * exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pdf


def muestrear_normal(mu: float, sigma: float, n: int, semilla: int = 0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.normal(mu, sigma, n)


def media_muestral(muestras: np.ndarray) -> float:
    return float(np.mean(muestras))


def varianza_muestral(muestras: np.ndarray) -> float:
    return float(np.var(muestras, ddof=0))


def estimar_media_con_ic(muestras: np.ndarray, confianza: float = 0.95) -> tuple[float, float]:
    """Devuelve (media, margen) usando distribución normal."""
    from scipy import stats
    n = len(muestras)
    se = np.std(muestras, ddof=1) / np.sqrt(n)
    h = stats.norm.ppf((1 + confianza) / 2) * se
    return float(np.mean(muestras)), float(h)


def main() -> int:
    rng = np.random.default_rng(42)
    n = 10000
    muestras = muestrear_normal(mu=5.0, sigma=2.0, n=n, semilla=42)
    print(f"Muestreo N(5, 2) con n={n}")
    print(f"  Media muestral: {media_muestral(muestras):.3f} (esperado 5.0)")
    print(f"  Varianza muestral: {varianza_muestral(muestras):.3f} (esperado 4.0)")
    # PMF binomial
    b = binomial(10, 0.5)
    print(f"  P(X=5 | Binomial(10, 0.5)) = {b(5):.3f}")
    # Bernoulli
    ber = bernoulli(0.3)
    print(f"  P(0 | Bern(0.3)) = {ber(0):.3f}, P(1) = {ber(1):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
