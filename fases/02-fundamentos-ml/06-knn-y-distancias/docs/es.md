# KNN y metricas de distancia

> KNN: el algoritmo de ML mas simple. Sin entrenamiento, sin parametros. Solo distancia y voto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-support-vector-machines
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar distancia euclidiana y KNN.
- Diagnosticar el parametro k.
- Estandarizar features antes de KNN.

## Constrúyelo

```python
import numpy as np


def euclidiana(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for x in X_test:
        dists = np.array([euclidiana(x, xt) for xt in X_train])
        idx = np.argsort(dists)[:k]
        vecinos = y_train[idx]
        valores, counts = np.unique(vecinos, return_counts=True)
        y_pred.append(valores[np.argmax(counts)])
    return np.array(y_pred)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-knn-tuning
fase: 02
leccion: 06
---

1. Accuracy bajo: normalizar features.
2. k=1 sobreajuste: k>1.
3. >10K: BallTree o KDTree.
4. Features irrelevantes: eliminarlas.
```

## Ejercicios

1. **KNN con pesos**: cada vecino vota con peso 1/distancia.
2. **KDTree**: usa sklearn.neighbors.KDTree para queries rapidos.
3. **Desafio**: implementa KNN con validacion leave-one-out.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 2
- scikit-learn KNeighborsClassifier: <https://scikit-learn.org/stable/modules/neighbors.html>

---

> 📚 **Adaptación al español** de la lección "[KNN and Distance Metrics]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).