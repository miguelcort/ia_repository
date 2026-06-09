# Feature engineering y seleccion

> La mayoria del tiempo en un proyecto de ML se gasta en feature engineering. No en el modelo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar estandarizacion, min-max, one-hot, polynomial features.
- Diagnosticar cuando aplicar cada transformacion.
- Distinguir feature engineering manual vs embeddings aprendidos.

## Constrúyelo

```python
import numpy as np


def estandarizar(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma


def one_hot(y, n_clases=None):
    if n_clases is None:
        n_clases = int(y.max() + 1)
    out = np.zeros((len(y), n_clases), dtype=int)
    out[np.arange(len(y)), y] = 1
    return out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-feat-eng
fase: 02
leccion: 08
---

1. Subajuste: polynomial features.
2. Sobreajuste: PCA o L1.
3. Distintas magnitudes: estandarizar.
4. Categoricas: target encoding.
```

## Ejercicios

1. **Target encoding**: codifica una categoria con la media del
   target.
2. **Bucketizar**: convierte una variable continua en categorica
   por cuantiles.
3. **Desafio**: implementa L1 (Lasso) feature selection.

## Lecturas recomendadas

- "Feature Engineering for Machine Learning" (Zheng & Casari)
- scikit-learn preprocessing: <https://scikit-learn.org/stable/modules/preprocessing.html>

---

> 📚 **Adaptación al español** de la lección "[Feature Engineering and Selection]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).