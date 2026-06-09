"""
Lección: 15-estadistica-para-ml
Fase: 01
Prerrequisitos: 06-probabilidad-y-distribuciones
"""
from __future__ import annotations
import sys
import numpy as np

try:
    from scipy import stats as st
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def media(x):
    return float(np.mean(x))


def mediana(x):
    return float(np.median(x))


def varianza(x, ddof=0):
    return float(np.var(x, ddof=ddof))


def std(x, ddof=0):
    return float(np.std(x, ddof=ddof))


def correlacion_pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if not HAS_SCIPY:
        # Fallback: ranking de Pearson
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return correlacion_pearson(rx.astype(float), ry.astype(float))
    return float(st.spearmanr(x, y).correlation)


def t_test_una_muestra(x, mu=0.0):
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    result = st.ttest_1samp(x, mu)
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 2.0, 100)
    y = x + rng.normal(0, 0.5, 100)
    print(f"Media: {media(x):.3f} (esperado ~5)")
    print(f"Mediana: {mediana(x):.3f}")
    print(f"Std: {std(x):.3f} (esperado ~2)")
    print(f"Pearson(x,y): {correlacion_pearson(x, y):.3f} (esperado alto)")
    t, p = t_test_una_muestra(x, mu=5.0)
    print(f"t-test contra mu=5: t={t:.3f}, p={p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())