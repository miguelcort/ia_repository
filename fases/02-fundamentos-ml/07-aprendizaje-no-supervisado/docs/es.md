# Aprendizaje no supervisado: K-Means, DBSCAN

> Sin etiquetas, el algoritmo encuentra estructura por si solo. K-Means es el mas simple y rapido.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-knn-y-distancias
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar K-Means desde cero.
- Diagnosticar el numero optimo de clusters.
- Conocer DBSCAN y sus diferencias con K-Means.

## Constrúyelo

```python
import numpy as np


def kmeans(X, k, max_iter=100, semilla=0):
    rng = np.random.default_rng(semilla)
    centroides = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(max_iter):
        dists = np.array([[np.linalg.norm(x - c) for c in centroides] for x in X])
        asignaciones = np.argmin(dists, axis=1)
        nuevos = np.array([X[asignaciones == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroides, nuevos, atol=1e-6):
            break
        centroides = nuevos
    return centroides, asignaciones
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-clustering-elegir
fase: 02
leccion: 07
---

1. Esfericos: K-Means.
2. Arbitrarios: DBSCAN.
3. Jerarquicos: aglomerativo.
4. K desconocido: DBSCAN.
```

## Ejercicios

1. **DBSCAN**: implementa DBSCAN basico.
2. **Silhouette score**: calcula cohesion vs separacion.
3. **Desafio**: implementa hierarchical clustering con linkage.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" cap. 12
- scikit-learn clustering: <https://scikit-learn.org/stable/modules/clustering.html>

---

> 📚 **Adaptación al español** de la lección "[Unsupervised Learning: K-Means, DBSCAN]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).