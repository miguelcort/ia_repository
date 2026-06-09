# Metodos de muestreo

> Cuando tu poblacion es enorme (todo internet), no puedes medirla toda. Muestras.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar muestreo aleatorio simple, con reemplazo, estratificado.
- Aplicar bootstrap para intervalos de confianza.
- Diagnosticar cuando cada metodo es apropiado.

## Constrúyelo

```python
from __future__ import annotations
import numpy as np


def bootstrap(x, estadistico, n_remuestras=1000, semilla=0):
    rng = np.random.default_rng(semilla)
    x = np.asarray(x)
    muestras = rng.choice(x, size=(n_remuestras, len(x)), replace=True)
    stats = np.array([estadistico(m) for m in muestras])
    return {
        "estimacion": float(estadistico(x)),
        "ic_95": (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))),
    }
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

1. Para IC: bootstrap.
2. Para varianza: estratificado o k-fold.
3. Para simulaciones: Monte Carlo de la distribucion.
4. Advertir contra data leakage.
```

## Ejercicios

1. **MCMC**: implementa Metropolis-Hastings para muestrear de
   una distribucion arbitraria.
2. **Stratified k-fold**: implementa cross-validation estratificada
   para clases desbalanceadas.
3. **Desafio**: implementa muestreo por rechazo de una distribucion
   arbitraria.

## Lecturas recomendadas

- "An Introduction to the Bootstrap" (Efron & Tibshirani)
- numpy.random: <https://numpy.org/doc/stable/reference/random/index.html>

---

> 📚 **Adaptación al español** de la lección "[Sampling Methods]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).