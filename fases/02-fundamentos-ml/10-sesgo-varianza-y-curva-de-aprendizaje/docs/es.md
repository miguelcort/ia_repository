# Sesgo, varianza y curva de aprendizaje

> El error total de un modelo se descompone en sesgo, varianza y ruido irreducible. Saber leer este tradeoff es la diferencia entre un modelo que mejora y uno que no.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 09-metricas-y-validacion-cruzada
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Estimar sesgo y varianza via bootstrap.
- Implementar curva de aprendizaje.
- Diagnosticar overfitting vs underfitting desde la curva.

## Constrúyelo

```python
import numpy as np


def curva_aprendizaje(modelo_fn, X, y, k=5):
    n = len(X)
    sizes = np.linspace(10, n, k, dtype=int)
    train_scores, val_scores = [], []
    for s in sizes:
        Xtr, ytr = X[:s], y[:s]
        Xte, yte = X[s:], y[s:]
        if len(Xte) == 0:
            break
        y_pred = modelo_fn(Xtr, ytr, Xte)
        val_scores.append(1 - np.mean((yte - y_pred) ** 2))
    return sizes[:len(val_scores)], val_scores
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bias-var
fase: 02
leccion: 10
---

1. Bajo+gap pequeno: underfitting. Modelo mas complejo.
2. Alto+val bajo: overfitting. Regularizacion.
3. Ambos altos: bien.
4. Plana: agregar features.
```

## Ejercicios

1. **Sesgo-varianza**: para un polinomio de grado 1, 5, 20,
   calcula sesgo y varianza via bootstrap.
2. **Curva de validacion**: agrega el error de validacion como
   segunda curva.
3. **Desafio**: implementa el decomposition sesgo^2 + varianza +
   ruido y verifica la igualdad.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 2
- "Understanding the Bias-Variance Tradeoff" (Scott Fortmann-Roe)

---

> 📚 **Adaptación al español** de la lección "[Bias, Variance and the Learning Curve]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).