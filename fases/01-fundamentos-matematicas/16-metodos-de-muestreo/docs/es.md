# 16 — Métodos de muestreo

> Cuando tu población es enorme (todo internet), no puedes medirla toda. Muestras.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar muestreo aleatorio simple, con reemplazo y
  estratificado.
- Aplicar bootstrap para intervalos de confianza de cualquier
  estadístico.
- Diagnosticar cuándo cada método es apropiado y cuándo
  introduce sesgo.
- Conocer MCMC (Metropolis-Hastings) y muestreo por rechazo
  como alternativas a muestreo directo.

## El problema

Si tu población es 8 mil millones de humanos, no puedes
entrevistar a todos. Tomas una muestra y extrapolas. El
problema es que el error de muestreo puede invalidar la
extrapolación si la muestra no es representativa. La lección
cubre los métodos clásicos de muestreo y bootstrap, que es la
herramienta de referencia para estimar varianza de cualquier
métrica en producción.

## El concepto

**Muestreo aleatorio simple (MAS).** Cada elemento de la
población tiene la misma probabilidad de ser elegido. Es el
método más simple pero no siempre el más eficiente: si tu
población tiene estratos obvios (por ejemplo, 50% mujeres y
50% hombres), un MAS de tamaño 100 puede terminar con 30/70 y
meter varianza innecesaria.

**Muestreo estratificado.** Divides la población en estratos
homogéneos (género, región, edad) y muestreas igual número de
cada estrato. Reduce la varianza del estimador a cambio de
conocer la estructura de la población. Es lo que hacen las
encuestas de opinión y los *evals* de LLM (por dominio,
idioma, etc.).

**Muestreo con reemplazo vs sin reemplazo.** Con reemplazo
cada elemento puede aparecer varias veces en la muestra; sin
reemplazo aparece a lo sumo una vez. El muestreo con reemplazo
es la base de bootstrap y de la mayoría de métodos de ML
(batch sampling, bagging).

**Bootstrap.** Remuestreo con reemplazo de la muestra original.
Si tienes una muestra `x` de tamaño `N` y quieres estimar la
distribución de un estadístico `T(x)`:

1. Por `b = 1, ..., B`: saca `x_b` de tamaño `N` con reemplazo
   de `x`.
2. Calcula `T(x_b)`.
3. La distribución empírica de los `T(x_b)` aproxima la
   distribución de muestreo de `T`.

Los percentiles 2.5 y 97.5 te dan el IC al 95%. Es universal:
no asume normalidad, no asume nada sobre la distribución
subyacente. Es computacionalmente caro pero trivialmente
paralelizable.

**Muestreo por rechazo.** Para muestrear de una distribución
`p(x)` compleja: encuentra una distribución `q(x)` fácil de
muestrear (la *proposal*) y una constante `M` tal que
`p(x) <= M * q(x)` para todo `x`. Muestrea `x ~ q` y acepta con
probabilidad `p(x) / (M * q(x))`. Si rechazas, intentas de
nuevo. Funciona en baja dimensión; en alta dimensión el
rechazo es exponencial.

**MCMC (Markov Chain Monte Carlo).** Construye una cadena de
Markov cuya distribución estacionaria es la que quieres
muestrear. Metropolis-Hastings es el algoritmo clásico: propone
un nuevo estado, acepta con probabilidad
`min(1, p(x') q(x|x') / (p(x) q(x'|x)))`. Después de suficientes
iteraciones, los estados muestreados aproximan `p(x)`. Es la
base de inferencia bayesiana y de muchos modelos generativos.

## Constrúyelo

```python
from __future__ import annotations
import numpy as np


def bootstrap(x, estadistico, n_remuestras=1000, semilla=0, alpha=0.05):
    rng = np.random.default_rng(semilla)
    x = np.asarray(x)
    n = len(x)
    muestras = rng.choice(x, size=(n_remuestras, n), replace=True)
    stats = np.array([estadistico(m) for m in muestras])
    low = (alpha / 2) * 100
    high = (1 - alpha / 2) * 100
    return {
        "estimacion": float(estadistico(x)),
        "ic": (
            float(np.percentile(stats, low)),
            float(np.percentile(stats, high)),
        ),
    }


def muestreo_estratificado(estratos: dict, n_por_estrato: int,
                          semilla: int = 0) -> dict:
    """Muestreo estratificado: n fijo por estrato."""
    rng = np.random.default_rng(semilla)
    return {
        k: rng.choice(v, size=n_por_estrato, replace=False)
        for k, v in estratos.items()
    }


def metropolis_hastings(log_p, x0, proposal_std, n_iter, semilla=0):
    """Muestrea de p(x) usando Metropolis-Hastings.
    log_p: función que devuelve log p(x)."""
    rng = np.random.default_rng(semilla)
    x = x0
    muestras = [x]
    log_px = log_p(x)
    for _ in range(n_iter - 1):
        x_prop = x + rng.normal(0, proposal_std)
        log_px_prop = log_p(x_prop)
        if np.log(rng.random()) < log_px_prop - log_px:
            x = x_prop
            log_px = log_px_prop
        muestras.append(x)
    return np.array(muestras)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sampling-elegir
fase: 01
leccion: 16
---

Eres un asistente que ayuda a elegir el método de muestreo
adecuado. Recibirás una descripción del problema (población,
estadístico de interés, restricciones). Tu trabajo:

1. Para IC de cualquier métrica: bootstrap.
2. Para varianza de un estimador: estratificado o k-fold.
3. Para simulaciones físicas: Monte Carlo de la distribución.
4. Para distribuciones complejas: MCMC o rechazo.
5. Advertir contra data leakage entre muestra y remuestra.
```

## Ejercicios

1. **MCMC**: implementa Metropolis-Hastings para muestrear de
   una distribución bimodal arbitraria.
2. **Stratified k-fold**: implementa cross-validation
   estratificada para clases desbalanceadas (mantener
   proporción de clases en cada fold).
3. **Desafío**: implementa muestreo por rechazo de una
   distribución arbitraria y compara con inverse-CDF cuando
   esté disponible.

## Lecturas recomendadas

- *An Introduction to the Bootstrap* — Efron & Tibshirani.
- *Sampling* — Steven K. Thompson (Wiley).
- numpy.random: <https://numpy.org/doc/stable/reference/random/index.html>.
- *Markov Chain Monte Carlo in Practice* — Gilks, Richardson,
  Spiegelhalter.

---

> 📚 **Adaptación al español** de la lección "[Sampling Methods]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
