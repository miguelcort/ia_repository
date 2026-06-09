"""
Lección: 16-deteccion-de-anomalias
Fase: 02
Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


def zscore_anomalias(x, umbral=3.0):
    """Detecta anomalias como |z-score| > umbral."""
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma == 0:
        return np.zeros(len(x), dtype=bool)
    z = np.abs((x - mu) / sigma)
    return z > umbral


def zscore_robusto(x, umbral=3.5):
    """Z-score modificado: usa mediana y MAD (median absolute deviation).
    Es robusto a outliers y los detecta mejor."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad == 0:
        return np.zeros(len(x), dtype=bool)
    z = 0.6745 * (x - med) / mad
    return np.abs(z) > umbral


def iqr_anomalias(x, factor=1.5):
    """Anomalias: fuera de [Q1 - factor*IQR, Q3 + factor*IQR]."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lim_inf = q1 - factor * iqr
    lim_sup = q3 + factor * iqr
    return (x < lim_inf) | (x > lim_sup)


def mahalanobis(X):
    """Distancia de Mahalanobis al centroide. Sirve para datos multivariados."""
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    cov += np.eye(cov.shape[0]) * 1e-9
    inv_cov = np.linalg.inv(cov)
    diff = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))


def mahalanobis_anomalias(X, umbral=None, percentil=97.5):
    """Anomalias: distancia de Mahalanobis > percentil."""
    dist = mahalanobis(X)
    if umbral is None:
        umbral = np.percentile(dist, percentil)
    return dist > umbral, dist


def isolation_score_simple(X, n_arboles=20, muestra_frac=0.5, semilla=0):
    """Isolation forest simplificado: depth promedio por muestra.
    Anomalias: depth menor (aisladas mas rapido)."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    depths = np.zeros(n)
    for _ in range(n_arboles):
        idx = rng.choice(n, size=int(n * muestra_frac), replace=False)
        X_sub = X[idx]
        # Calcula distancia minima al vecino mas cercano en subgrupo
        # (proxy rapido: nearest neighbor)
        from itertools import combinations
        for i_local, i in enumerate(idx):
            if i_local == 0:
                depths[i] += 0
                continue
            dists = np.linalg.norm(X_sub[:i_local] - X_sub[i_local], axis=1)
            depths[i] += dists.min() if len(dists) > 0 else 0
    return depths / n_arboles


def main() -> int:
    rng = np.random.default_rng(0)
    # Serie con outliers
    x = np.concatenate([rng.normal(size=100), np.array([10.0, -10.0, 12.0])])
    z_anom = zscore_anomalias(x, umbral=3.0)
    zr_anom = zscore_robusto(x, umbral=3.5)
    iqr_anom = iqr_anomalias(x)
    print(f"Z-score anomalias: {z_anom.sum()}")
    print(f"Z-score robusto anomalias: {zr_anom.sum()}")
    print(f"IQR anomalias: {iqr_anom.sum()}")
    # Mahalanobis multivariado
    X = rng.normal(size=(100, 2))
    X[0] = [5, 5]  # outlier
    anom, dist = mahalanobis_anomalias(X, percentil=95)
    print(f"Mahalanobis anomalias: {anom.sum()}")
    print(f"Distancia[0]: {dist[0]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())