# 15 — Estadística para ML

> La mayoría de papers de ML no reportan tests estadísticos. Eso es un problema.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular media, mediana, varianza, desviación estándar y
  correlación de Pearson desde cero y con NumPy.
- Aplicar t-test de una muestra y de muestras pareadas desde
  cero y con SciPy.
- Diagnosticar cuándo un test paramétrico vs no paramétrico es
  apropiado.
- Implementar bootstrap para intervalos de confianza sin
  bibliotecas externas.

## El problema

Si comparas dos modelos en un solo split de test, no sabes si
la diferencia de accuracy es real o ruido. Un paper de NeurIPS
2024 que reporta "93.2% vs 92.8% en un test set" sin intervalos
de confianza está contando la mitad de la historia. La
estadística inferencial existe para responder exactamente la
pregunta "¿esta diferencia es real?".

La lección cubre lo mínimo indispensable: estadísticos
descriptivos, t-test, correlación de Pearson y Spearman,
bootstrap. No es un curso de estadística: es un manual de
supervivencia para que el estudiante no publique resultados que
no se sostienen.

## El concepto

**Estadísticos descriptivos.** Media, mediana, varianza y std
son las cuatro medidas que resumen un array. La media es
sensible a outliers; la mediana no. La varianza mide dispersión
cuadrática; la std es su raíz. Importante: NumPy `np.std` por
defecto usa `ddof=0` (divisor `N`); SciPy y la convención
*sample* usan `ddof=1` (divisor `N-1`). La diferencia es
irrelevante para `N>1000` y significativa para `N<30`.

**Correlación de Pearson.** Mide la fuerza de la relación
**lineal** entre dos variables. Devuelve un valor en `[-1, 1]`.
Una correlación alta no implica causalidad: el clásico ejemplo
es la correlación entre ventas de helado y ahogamientos, ambas
causadas por el calor del verano.

**Correlación de Spearman.** Mide la fuerza de la relación
**monotónica** (no necesariamente lineal). Equivale a calcular
Pearson sobre los rangos. Úsala cuando la relación es
monotónica pero no lineal, o cuando hay outliers que distorsionan
Pearson.

**t-test de una muestra.** Compara la media de una muestra con
un valor de referencia. Hipótesis nula: la media es igual a
`mu`. Reporta estadístico `t` y `p-value`. Si `p < 0.05`,
rechazas la nula al 5%.

**t-test pareado.** Compara dos conjuntos de measurements
pareados (mismo sujeto, dos condiciones). Es más potente que el
t-test de dos muestras independientes cuando hay correlación
pareada. En ML: comparar dos modelos sobre los mismos folds.

**Bootstrap.** Remuestreo con reemplazo para estimar la
distribución de un estadístico. Si tienes una muestra `x` de
tamaño `N`, tomas `B` remuestras de tamaño `N` con reemplazo,
calculas el estadístico en cada una, y los percentiles 2.5 y
97.5 te dan el IC al 95%. Es universal: funciona para
cualquier estadístico (mediana, cuantiles, métricas asimétricas)
sin supuestos distribucionales.

**Cuándo usar cada test.**

| Comparación | Test |
|---|---|
| 2 modelos, mismo test set | t-test pareado o Wilcoxon |
| 3+ modelos, mismo test set | ANOVA o Friedman |
| Relación lineal | Pearson |
| Relación monotónica | Spearman |
| ¿La muestra viene de distribución X? | KS o chi-cuadrado |
| Intervalo de confianza de cualquier métrica | Bootstrap |

## Constrúyelo

```python
from __future__ import annotations
import numpy as np
from scipy import stats as st


def media(x):
    return float(np.mean(x))


def mediana(x):
    return float(np.median(x))


def varianza(x, ddof=1):
    return float(np.var(x, ddof=ddof))


def std(x, ddof=1):
    return float(np.std(x, ddof=ddof))


def correlacion_pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return float(st.spearmanr(x, y).correlation)


def t_test_una_muestra(x, mu=0.0):
    r = st.ttest_1samp(x, mu)
    return float(r.statistic), float(r.pvalue)


def bootstrap(x, estadistico, n_remuestras=1000, semilla=0, alpha=0.05):
    """Intervalo de confianza bootstrap al (1-alpha)*100%."""
    rng = np.random.default_rng(semilla)
    x = np.asarray(x)
    n = len(x)
    muestras = rng.choice(x, size=(n_remuestas, n), replace=True)
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
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-test-elegir
description: Elegir el test estadístico adecuado
fase: 01
leccion: 15
---

Eres un asistente que ayuda a elegir el test estadístico
adecuado. Recibirás una descripción del experimento (qué se
compara, cuántas muestras, tipo de variable). Tu trabajo:

1. Compara 2 modelos: t-test pareado o Wilcoxon.
2. Compara 3+ modelos: ANOVA o Friedman.
3. Correlación: Pearson (lineal) o Spearman (monótona).
4. Distribuciones: KS o chi-cuadrado.
5. Advertir contra comparar un solo split.
6. Recomendar bootstrap para IC de cualquier métrica.
```

## Ejercicios

1. **Bootstrap**: implementa un intervalo de confianza del 95%
   para la mediana via bootstrap (N=1000 remuestreos).
2. **Test de permutación**: implementa el test de permutación
   para comparar dos accuracy (más robusto que t-test cuando
   las suposiciones no se cumplen).
3. **Desafío**: implementa la corrección de Bonferroni para
   múltiples testing (ajusta el umbral de significancia según
   el número de hipótesis).

## Lecturas recomendadas

- *Statistics for High-Dimensional Data* — Bühlmann & van de
  Geer.
- *An Introduction to the Bootstrap* — Efron & Tibshirani.
- scipy.stats: <https://docs.scipy.org/doc/scipy/reference/stats.html>.
- *Think Stats* — Allen Downey, libro libre en línea.

---

> 📚 **Adaptación al español** de la lección "[Statistics for ML]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
