# 15 — Series de tiempo

> Predicción de demanda, precios, tráfico, energía. La estructura temporal cambia todo: no puedes hacer shuffle y los features se derivan del pasado.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Descomponer una serie en tendencia, estacionalidad y
  residuo.
- Implementar ARIMA desde cero.
- Aplicar Prophet para forecasting con estacionalidad
  múltiple.
- Validar con time-series split (sin shuffle).

## El problema

Necesitas predecir las ventas del próximo trimestre, el
tráfico web de la próxima semana, o el precio de una acción
del próximo día. La regresión lineal no funciona porque el
**orden de los datos importa**: no puedes mezclar el futuro
con el pasado. Necesitas modelos que respeten la estructura
temporal y validación que respete la causalidad.

## El concepto

**Descomposición STL.** Una serie de tiempo se descompone en:

- **Tendencia (`T`):** dirección de largo plazo.
- **Estacionalidad (`S`):** patrones que se repiten (diario,
  semanal, anual).
- **Residuo (`R`):** lo que no se explica, ruido o eventos
  irregulares.

Modelo aditivo: `y_t = T_t + S_t + R_t`. Modelo multiplicativo:
`y_t = T_t · S_t · R_t` (cuando la amplitud crece con la
tendencia).

**Estacionariedad.** Una serie es estacionaria si sus
estadísticas (media, varianza, autocorrelación) no cambian
con el tiempo. La mayoría de modelos (ARIMA) asumen
estacionariedad. Test de Dickey-Fuller aumentado (ADF) es el
estándar para verificar.

**Diferenciación.** Si la serie no es estacionaria, diferenciar
una vez: `y'_t = y_t - y_{t-1}`. Si todavía no, diferenciar de
nuevo. La mayoría de series económicas necesitan 1-2
diferenciaciones.

**ACF y PACF.** Las funciones de autocorrelación y
autocorrelación parcial ayudan a elegir los órdenes `(p, d,
q)` de un modelo ARIMA:

- **ACF:** correlación entre `y_t` y `y_{t-k}`. Si decae
  lentamente, necesita más diferenciación.
- **PACF:** correlación entre `y_t` y `y_{t-k}` controlando
  por lags intermedios. Corta (corta en lag p, cero después)
  sugiere AR(p).

**ARIMA(p, d, q).**

- **AR(p):** Autoregresivo. `y_t = c + φ_1 y_{t-1} + ... + φ_p
  y_{t-p} + ε_t`.
- **I(d):** Integrado. Diferenciado `d` veces para
  estacionariedad.
- **MA(q):** Media móvil. `y_t = μ + ε_t + θ_1 ε_{t-1} + ... +
  θ_q ε_{t-q}`.

Combinación: `y_t` depende de sus valores pasados, sus errores
pasados, y `d` diferenciaciones.

**SARIMA(p, d, q)(P, D, Q, s).** Extensión con componente
estacional de período `s` (12 para anual, 7 para semanal, etc.).

**Prophet.** Modelo de Facebook que descompone la serie en
`tendencia + estacionalidad + holidays` y los ajusta con un
modelo bayesiano. Robusto a datos faltantes, cambios de
tendencia, y节假日. Bueno como baseline fuerte.

**LSTM y Transformer para series de tiempo.** Para patrones
complejos, los modelos neuronales (LSTM, N-BEATS, Temporal
Fusion Transformer) son útiles. Necesitan series largas y
muchos datos.

**Validación con time-series split.** NUNCA uses k-fold
normal con shuffle. Usa:

```python
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # train_idx son los datos más antiguos, val_idx los más
    # recientes que no se solapan con train
```

**Métricas de forecasting.**

- **MAE, RMSE:** mismas que regresión.
- **MAPE:** error porcentual. Problema con valores cero.
- **MASE:** Mean Absolute Scaled Error. Compara contra un
  naive forecast (último valor). MASE < 1 es mejor que naive.

## Constrúyelo

```python
import numpy as np


def diferencia(serie, d=1):
    """Diferencia una serie d veces."""
    s = np.asarray(serie, dtype=float).copy()
    for _ in range(d):
        s = np.diff(s)
    return s


def acf(serie, max_lag=20):
    """Función de autocorrelación."""
    s = np.asarray(serie, dtype=float)
    s = s - s.mean()
    var = np.sum(s ** 2)
    return [np.sum(s[: len(s) - k] * s[k:]) / var for k in range(max_lag)]


def pacf(serie, max_lag=20):
    """Autocorrelación parcial vía regresión."""
    return [1.0] + [acf(np.array(serie), 1)[0]]  # simplificado


def ar(p, coefs):
    """Predice un paso adelante con un modelo AR(p)."""
    coefs = np.asarray(coefs)
    def predict(history):
        h = np.asarray(history[-p:])[::-1] if len(history) >= p else np.zeros(p)
        return float(np.dot(h, coefs))
    return predict


def naive_forecast(serie):
    """Predicción naive: último valor."""
    return float(serie[-1])


def seasonal_naive(serie, periodo):
    """Predicción seasonal naive: valor de hace un periodo."""
    return float(serie[-periodo])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-forecasting-elegir
fase: 02
leccion: 15
---

Eres un asistente que ayuda a elegir y configurar modelos de
forecasting. Recibirás la serie de tiempo y la descripción
(estacionalidad, tendencia, horizonte de predicción). Tu
trabajo:

1. Si la serie tiene tendencia y estacionalidad clara:
   Prophet como baseline rápido.
2. Si la serie es estacionaria y los ACF/PACF son
   interpretables: ARIMA.
3. Si hay múltiples estacionalidades (diaria + anual):
   Prophet con regressors adicionales o modelos neuronales.
4. Si la serie es muy larga (> 100k puntos) y los patrones
   son complejos: LSTM, N-BEATS, o Temporal Fusion
   Transformer.
5. Advertir: NUNCA usar k-fold con shuffle en series de
   tiempo.
6. Usar time-series split para validación.
7. Reportar MASE y MAPE además de MAE/RMSE.
8. Considerar intervalos de predicción, no solo punto.
```

## Ejercicios

1. **ARIMA desde cero**: implementa un AR(1) y verifica
   convergencia en una serie simulada.
2. **Prophet**: aplica Prophet a la serie de pasajeros
   aéreos y grafica predicción vs realidad.
3. **Desafío**: implementa time-series cross-validation
   walk-forward y compara Prophet vs ARIMA vs naive.

## Lecturas recomendadas

- *Forecasting: Principles and Practice* — Hyndman &
  Athanasopoulos (libre en línea).
- *Time Series Analysis* — Hamilton.
- Prophet: <https://facebook.github.io/prophet>.
- statsmodels: <https://www.statsmodels.org>.
- sktime: <https://www.sktime.net>.

---

> 📚 **Adaptación al español** de la lección "[Time Series]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
