# Arboles de decision y Random Forest

> Los arboles son el unico modelo de ML que puedes explicar a tu abuela. RF es un ensemble de arboles.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-que-es-machine-learning
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar entropia y ganancia de informacion.
- Encontrar el mejor split por fuerza bruta.
- Diagnosticar profundidad optima y overfitting.

## Constrúyelo

```python
import numpy as np


def entropia(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def ganancia_informacion(y, izq, der):
    n = len(y)
    h = entropia(y)
    h_ponderada = (len(izq) * entropia(y[izq]) + len(der) * entropia(y[der])) / n
    return h - h_ponderada
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tree-depth
fase: 02
leccion: 04
---

1. Overfit: limitar max_depth.
2. Underfit: aumentar profundidad.
3. Lento: menos estimators.
4. Importances desiguales: seleccion de features.
```

## Ejercicios

1. **Decision tree completo**: implementa recursion para construir
   un arbol con max_depth.
2. **Random Forest**: implementa N arboles con bootstrap.
3. **Desafio**: implementa feature importance usando permutation
   importance.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 8
- scikit-learn DecisionTreeClassifier: <https://scikit-learn.org/stable/modules/tree.html>

---

> 📚 **Adaptación al español** de la lección "[Decision Trees and Random Forest]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).