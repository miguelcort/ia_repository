"""
Lección: 16-metodos-de-muestreo
Fase: 01
Prerrequisitos: 06-probabilidad-y-distribuciones
"""
from __future__ import annotations
import sys
import numpy as np


def muestreo_aleatorio_simple(poblacion, n, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(poblacion), size=n, replace=False)
    return poblacion[idx]


def muestreo_aleatorio_con_reemplazo(poblacion, n, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(poblacion), size=n, replace=True)
    return poblacion[idx]


def muestreo_estratificado(poblacion, estratos, n_por_estrato, semilla=0):
    rng = np.random.default_rng(semilla)
    resultado = []
    for e, k in zip(estratos, n_por_estrato):
        idx = rng.choice(e, size=k, replace=False)
        resultado.extend(poblacion[idx])
    return np.array(resultado)


def bootstrap(x, estadistico, n_remuestras=1000, semilla=0):
    """Estimacion del estadistico y su intervalo de confianza via bootstrap."""
    rng = np.random.default_rng(semilla)
    x = np.asarray(x)
    muestras = rng.choice(x, size=(n_remuestras, len(x)), replace=True)
    stats = np.array([estadistico(m) for m in muestras])
    return {
        "estimacion": float(estadistico(x)),
        "media_boot": float(np.mean(stats)),
        "std_boot": float(np.std(stats)),
        "ic_95": (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))),
    }


def main() -> int:
    poblacion = np.arange(1000)
    muestra = muestreo_aleatorio_simple(poblacion, 50)
    print(f"Muestreo simple n=50: media={muestra.mean():.2f}")
    # Bootstrap
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 100)
    res = bootstrap(x, np.mean, n_remuestras=500)
    print(f"Bootstrap media={res['media_boot']:.3f}, "
          f"IC95=({res['ic_95'][0]:.3f}, {res['ic_95'][1]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())