# Metricas y validacion cruzada

> Accuracy es la metrica mas comun pero la menos informativa. Saber cual usar requiere entender el problema.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero, 03-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar accuracy, precision, recall, F1.
- Generar K-fold cross-validation.
- Diagnosticar cuando cada metrica es apropriada.

## Constrúyelo

```python
import numpy as np


def accuracy(y_v, y_p):
    return float(np.mean(y_v == y_p))


def f1(y_v, y_p, clase=1):
    tp = np.sum((y_v == clase) & (y_p == clase))
    fp = np.sum((y_v != clase) & (y_p == clase))
    fn = np.sum((y_v == clase) & (y_p != clase))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-metric-elegir
fase: 02
leccion: 09
---

1. Balanceada: accuracy.
2. Desbalanceada: F1, AUC-PR.
3. Regresion: MSE, MAE, R^2.
4. Critico: recall.
```

## Ejercicios

1. **AUC-ROC**: implementa el area bajo la curva ROC.
2. **Confusion matrix**: implementa la matriz KxK.
3. **Desafio**: implementa stratified K-fold.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 2, 5
- sklearn metrics: <https://scikit-learn.org/stable/modules/model_evaluation.html>

---

> 📚 **Adaptación al español** de la lección "[Metrics and Cross-Validation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).