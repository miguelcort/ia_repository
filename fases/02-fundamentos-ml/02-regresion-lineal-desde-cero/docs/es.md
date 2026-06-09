# Regresion lineal desde cero

> Regresion lineal es el 'hello world' de ML. Simple, interpretable, y aun sorprendentemente util en datos tabulares.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-que-es-machine-learning
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar minimos cuadrados por forma cerrada.
- Evaluar con MSE y R^2.
- Diagnosticar multicolinealidad y regularizar.

## Constrúyelo

```python
import numpy as np


def ajustar(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ y)


def predecir(w, X):
    X = np.asarray(X, dtype=float)
    n = len(X)
    X_aug = np.column_stack([np.ones(n), X])
    return X_aug @ w


def r2(y_v, y_p):
    ss_res = np.sum((y_v - y_p) ** 2)
    ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-linear-reg
fase: 02
leccion: 02
---

1. Si LinAlgError, usar lstsq o ridge.
2. Si R^2 bajo, features faltantes o no lineal.
3. Coeficientes enormes: multicolinealidad.
4. >100K features: SGD.
```

## Ejercicios

1. **Ridge regression**: implementa w = (X^T X + lambda I)^-1 X^T y.
2. **Lasso**: implementa con soft-thresholding.
3. **Desafio**: implementa regresion polinomica de grado 3 con
   regularizacion.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 3 y 6
- scikit-learn LinearRegression: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>

---

> 📚 **Adaptación al español** de la lección "[Linear Regression from Scratch]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).