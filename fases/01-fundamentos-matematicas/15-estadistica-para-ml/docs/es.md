# Estadistica para ML

> La mayoria de papers de ML no reportan tests estadisticos. Eso es un problema.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular media, mediana, varianza, std, correlacion.
- Aplicar t-test y Spearman desde cero.
- Diagnosticar cuando un test es apropiado.

## Constrúyelo

```python
from __future__ import annotations
import sys
import numpy as np
from scipy import stats as st

def media(x): return float(np.mean(x))
def mediana(x): return float(np.median(x))
def std(x): return float(np.std(x))
def correlacion_pearson(x, y): return float(np.corrcoef(x, y)[0, 1])
def t_test_una_muestra(x, mu=0.0): return float(st.ttest_1samp(x, mu).pvalue)
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
description: Elegir el test estadistico adecuado
fase: 01
leccion: 15
---

1. Comparar 2 modelos: t-test pareado o Wilcoxon.
2. Comparar 3+ modelos: ANOVA o Friedman.
3. Correlacion: Pearson (lineal) o Spearman (monotona).
4. Distribuciones: KS o chi-cuadrado.
5. Advertir contra comparar un solo split.
```

## Ejercicios

1. **Bootstrap**: implementa un intervalo de confianza del 95%
   para la media via bootstrap (N=1000 remuestreos).
2. **Test de permutacion**: implementa el test de permutacion
   para comparar dos accuracy.
3. **Desafio**: implementa la correccion de Bonferroni para
   multiples testing.

## Lecturas recomendadas

- "Statistics for High-Dimensional Data" (Buhlmann & van de Geer)
- scipy.stats: <https://docs.scipy.org/doc/scipy/reference/stats.html>

---

> 📚 **Adaptación al español** de la lección "[Statistics for ML]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).