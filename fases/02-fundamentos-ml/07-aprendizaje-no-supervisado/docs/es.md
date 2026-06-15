# 07 — Aprendizaje no supervisado: K-Means y DBSCAN

> Cuando no tienes etiquetas, los algoritmos no supervisados descubren estructura. K-Means para clusters compactos, DBSCAN para clusters de forma arbitraria.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar K-Means desde cero y entender su convergencia.
- Aplicar el método del codo y silueta para elegir `k`.
- Implementar DBSCAN y entender su diferencia con K-Means.
- Aplicar reducción de dimensionalidad con PCA y t-SNE.

## El problema

Tienes datos de clientes (1M de usuarios) y quieres
segmentarlos para marketing. No tienes etiquetas: no sabes
de antemano qué segmentos existen. Los algoritmos de
*clustering* descubren la estructura. K-Means es el más usado
y funciona cuando los clusters son compactos y esféricos.
DBSCAN es la alternativa cuando los clusters tienen forma
arbitraria o hay ruido.

## El concepto

**K-Means.** Algoritmo iterativo que asigna cada punto al
centroide más cercano y luego actualiza los centroides como
el promedio de sus puntos asignados. Repite hasta convergencia.

Algoritmo:

1. Inicializa `k` centroides (aleatorio o K-Means++).
2. Repite hasta convergencia:
   - Asigna cada punto al centroide más cercano.
   - Recalcula cada centroide como la media de sus puntos
     asignados.

Converge en `O(n · k · iteraciones)`. K-Means++ (inicialización
inteligente) da resultados mucho mejores que la inicialización
aleatoria.

**Limitaciones de K-Means.**

- Asume clusters esféricos y de igual varianza.
- Sensible a outliers (un outlier jala su centroide).
- Requiere especificar `k` de antemano.
- Solo encuentra clusters convexos.

**Elegir `k`.**

- **Método del codo:** grafica la suma de distancias
  intra-cluster vs `k`. Busca el "codo" donde agregar más
  clusters no reduce mucho la suma.
- **Coeficiente de silueta:** para cada punto, mide qué tan
  similar es a su cluster vs al cluster más cercano. Promedio
  en `[-1, 1]`. Más alto es mejor.
- **Gap statistic:** compara la inercia con la esperada bajo
  una distribución nula.

**DBSCAN (Density-Based Spatial Clustering of Applications
with Noise).** Agrupa puntos que están densamente empaquetados
y marca como outliers los puntos en regiones de baja densidad.
No requiere especificar `k` y encuentra clusters de forma
arbitraria. Parámetros:

- `eps`: radio del vecindario.
- `min_samples`: mínimo de puntos para formar un cluster.

Un punto es **core** si tiene al menos `min_samples` puntos en
su `eps`-vecindario. Es **border** si está en el vecindario de
un core pero no es core. Es **noise** si no es ni core ni
border. DBSCAN conecta cores que estén dentro de `eps`.

**Cuándo usar K-Means vs DBSCAN.**

| Situación | K-Means | DBSCAN |
|---|---|---|
| Clusters esféricos y compactos | Sí | Sí |
| Clusters de forma arbitraria | No | Sí |
| Detección de outliers | No | Sí |
| Conocer `k` de antemano | Necesario | No necesario |
| Datos muy grandes | Sí (MiniBatchKMeans) | Limitado |

**Reducción de dimensionalidad para visualizar clusters.**

- **PCA** (linear, rápido): preserva varianza global.
- **t-SNE** (no linear, lento): preserva vecindad local.
  Bueno para visualización en 2D/3D.
- **UMAP** (no linear, más rápido que t-SNE): preserva tanto
  local como global.

**Métricas de evaluación de clustering** (cuando no hay
ground truth):

- **Silueta:** `[-1, 1]`, más alto mejor.
- **Davies-Bouldin:** menor mejor.
- **Calinski-Harabasz:** mayor mejor.

## Constrúyelo

```python
import numpy as np


def kmeans(X, k, max_iter=300, semilla=0, tol=1e-4):
    """K-Means con inicialización aleatoria."""
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    # Inicialización aleatoria
    idx = rng.choice(n, size=k, replace=False)
    centroides = X[idx].copy()
    for _ in range(max_iter):
        # Asignar cada punto al centroide más cercano
        dists = np.linalg.norm(
            X[:, None, :] - centroides[None, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)
        # Actualizar centroides
        nuevos = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.linalg.norm(nuevos - centroides) < tol:
            break
        centroides = nuevos
    return centroides, labels


def kmeans_plus_plus(X, k, semilla=0):
    """Inicialización K-Means++: mejor que aleatoria."""
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    centroides = [X[rng.integers(n)]]
    for _ in range(1, k):
        dists = np.min(
            np.linalg.norm(X[:, None] - np.array(centroides)[None, :], axis=2),
            axis=1,
        )
        probs = dists ** 2 / (dists ** 2).sum()
        idx = rng.choice(n, p=probs)
        centroides.append(X[idx])
    return np.array(centroides)


def dbscan(X, eps, min_samples):
    """DBSCAN: density-based clustering."""
    n = X.shape[0]
    labels = np.full(n, -1)  # -1 = no visitado / noise
    cluster_id = 0

    def vecinos(i):
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(n):
        if labels[i] != -1:
            continue
        seed = vecinos(i)
        if len(seed) < min_samples:
            labels[i] = -1  # noise (temporal)
            continue
        # Empezar nuevo cluster
        labels[i] = cluster_id
        seed = list(seed[seed != i])
        j = 0
        while j < len(seed):
            q = seed[j]
            if labels[q] == -1:
                labels[q] = cluster_id
            if labels[q] == -1 or labels[q] == cluster_id:
                if labels[q] == -1:
                    labels[q] = cluster_id
                new_vecinos = vecinos(q)
                if len(new_vecinos) >= min_samples:
                    seed.extend(
                        [v for v in new_vecinos if v not in seed]
                    )
            j += 1
        cluster_id += 1
    return labels
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

Eres un asistente que ayuda a elegir y configurar algoritmos de
clustering. Recibirás la descripción del problema (tamaño del
dataset, sospecha sobre la forma de los clusters, presencia de
outliers). Tu trabajo:

1. Si clusters compactos y esféricos: K-Means con k elegido
   por método del codo o silueta.
2. Si clusters de forma arbitraria o presencia de outliers:
   DBSCAN.
3. Si el dataset es muy grande: MiniBatchKMeans.
4. Si quieres visualizar en 2D/3D: PCA (rápido) o t-SNE
   (lento pero mejor para local).
5. Siempre: estandarizar features antes de K-Means.
6. Reportar silueta, Davies-Bouldin y Calinski-Harabasz.
7. Si sabes la verdad de tierra (ground truth), usar
   Adjusted Rand Index (ARI) o NMI.
```

## Ejercicios

1. **K-Means++**: implementa la inicialización K-Means++ y
   compara convergencia con inicialización aleatoria.
2. **DBSCAN vs K-Means**: aplica ambos a un dataset con
   clusters en forma de luna y compara.
3. **Desafío**: implementa clustering aglomerativo
   jerárquico y compara con K-Means en datasets sintéticos.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 12.
- scikit-learn clustering: <https://scikit-learn.org/stable/modules/clustering.html>.
- DBSCAN paper: Ester et al., 1996.
- UMAP: <https://umap-learn.readthedocs.io>.

---

> 📚 **Adaptación al español** de la lección "[Unsupervised Learning: K-Means, DBSCAN]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
