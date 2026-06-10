"""
Lección: 13-depuracion-de-redes-neuronales
Fase: 03
Prerrequisitos: 09-programacion-de-learning-rate
"""
from __future__ import annotations
import sys
import numpy as np


def check_gradiente_numerico(f, df, x, h=1e-5):
    """Verifica el gradiente numerico (diferencias finitas) vs el
    computado por df(x).
    diff = (f(x+h) - f(x-h)) / (2*h) - df(x).
    Si difieren, hay bug en df."""
    x = np.asarray(x, dtype=float)
    grad_analitico = np.asarray(df(x), dtype=float).reshape(x.shape)
    grad_num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        diff = (f(x_plus) - f(x_minus)) / (2 * h)
        if hasattr(diff, '__iter__'):
            grad_num[i] = float(diff.flatten()[0]) if diff.size > 0 else 0.0
        else:
            grad_num[i] = float(diff)
    return float(np.abs(grad_analitico - grad_num).max())


def check_nan_inf(arr, nombre="tensor"):
    """Detecta NaN o Inf en un tensor."""
    arr = np.asarray(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    if nan_count > 0 or inf_count > 0:
        return f"ERROR en {nombre}: {nan_count} NaN, {inf_count} Inf"
    return f"OK en {nombre}: sin NaN/Inf"


def monitorear_pesos(pesos_por_capa):
    """Imprime estadisticas de los pesos de cada capa.
    Sospechoso: media muy alta, std muy baja o muy alta."""
    reporte = []
    for i, W in enumerate(pesos_por_capa):
        mu = float(W.mean())
        std = float(W.std())
        max_abs = float(np.abs(W).max())
        cero_por = float((W == 0).sum() / W.size * 100)
        ok = "OK" if (abs(mu) < 0.1 and 0.01 < std < 1.0) else "REVISAR"
        reporte.append(f"Capa {i}: mu={mu:+.3f} std={std:.3f} max|W|={max_abs:.2f} ceros={cero_por:.1f}% [{ok}]")
    return "\n".join(reporte)


def grad_check_paso(f, df, x, h=1e-4):
    """Wrapper simple: devuelve un dict con resultados del gradient check."""
    gradiente = df(x)
    return {
        "diff_max": check_gradiente_numerico(f, df, x, h),
        "nan_inf": check_nan_inf(gradiente, "gradiente"),
        "ok": check_gradiente_numerico(f, df, x, h) < 1e-5,
    }


def main() -> int:
    # Gradient check: f(x) = x^2, df = 2x
    f = lambda x: x ** 2
    df = lambda x: 2 * x
    resultado = grad_check_paso(f, df, np.array([3.0]))
    print(f"Gradient check (x=3): {resultado}")
    # Pesos
    rng = np.random.default_rng(0)
    capas = [rng.normal(scale=0.1, size=(10, 5)) for _ in range(3)]
    print(monitorear_pesos(capas))
    return 0


if __name__ == "__main__":
    sys.exit(main())