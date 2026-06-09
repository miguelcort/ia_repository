"""
Lección: 15-series-de-tiempo
Fase: 02
Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


def media_movil(x, ventana):
    """Media movil simple."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(ventana - 1, n):
        out[i] = x[i - ventana + 1:i + 1].mean()
    return out


def diferencias(x, orden=1):
    """Aplica diferencias: y[t] = x[t] - x[t-orden]."""
    y = np.diff(x, n=orden)
    return y


def autocorrelation(x, lag_max=20):
    """Calcula autocorrelacion para lags 0..lag_max."""
    x = x - x.mean()
    var = (x * x).sum()
    resultado = []
    for k in range(lag_max + 1):
        if k == 0:
            r = 1.0
        else:
            r = (x[:-k] * x[k:]).sum() / var
        resultado.append(r)
    return np.array(resultado)


def ar_forecast(x, orden=1, horizonte=5):
    """Forecast naive: AR(1) con x[t] = c + phi * x[t-1]."""
    n = len(x)
    if n < 2:
        return np.full(horizonte, x[-1] if n > 0 else 0.0)
    # Ajuste: y = x[t], x_pred = x[t-1]
    y = x[1:]
    X_pred = x[:-1]
    # Minimos cuadrados: phi = sum(x_pred * y) / sum(x_pred^2)
    phi = float((X_pred * y).sum() / (X_pred * X_pred).sum())
    c = float(y.mean() - phi * X_pred.mean())
    # Forecast recursivo
    forecast = []
    ultimo = x[-1]
    for _ in range(horizonte):
        sig = c + phi * ultimo
        forecast.append(sig)
        ultimo = sig
    return np.array(forecast)


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> int:
    rng = np.random.default_rng(0)
    # Serie AR(1) sintetica: x[t] = 0.7 * x[t-1] + ruido
    n = 100
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    # Test
    ma = media_movil(x, ventana=5)
    print(f"Media movil (ultimos 5): {ma[-5:]}")
    diff = diferencias(x, orden=1)
    print(f"Diferencia (ultimos 5): {diff[-5:]}")
    acf = autocorrelation(x, lag_max=10)
    print(f"ACF lags 0..10: {acf}")
    forecast = ar_forecast(x, orden=1, horizonte=5)
    print(f"Forecast proximos 5: {forecast}")
    return 0


if __name__ == "__main__":
    sys.exit(main())