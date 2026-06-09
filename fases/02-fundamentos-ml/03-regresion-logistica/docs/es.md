# Regresion logistica y clasificacion

> Regresion logistica es el 'hello world' de clasificacion. Simple, interpretable, y base de redes neuronales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar sigmoid y regresion logistica.
- Ajustar por descenso por gradiente con cross-entropy.
- Diagnosticar accuracy y problemas de clasificacion.

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ajustar(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w = np.zeros(d + 1)
    X_aug = np.column_stack([np.ones(n), X])
    for _ in range(epochs):
        p = sigmoid(X_aug @ w)
        grad = X_aug.T @ (p - y) / n
        w -= lr * grad
    return w
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-classification
fase: 02
leccion: 03
---

1. Overfitting: regularizar.
2. Bajo: features o modelo.
3. Desbalance: class_weight o resampling.
4. Umbral: precision-recall.
```

## Ejercicios

1. **Multiclase**: implementa softmax + cross-entropy para 3+ clases.
2. **Regularizacion L2**: agrega ||w||^2 al gradiente.
3. **Desafio**: implementa regresion logistica regularizada con
   early stopping.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 4
- scikit-learn LogisticRegression: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>

---

> 📚 **Adaptación al español** de la lección "[Logistic Regression and Classification]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).