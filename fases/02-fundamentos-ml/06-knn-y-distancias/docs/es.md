# 06 — KNN y métricas de distancia

> KNN: el algoritmo de ML más simple. Sin entrenamiento, sin parámetros. Solo distancia y voto.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-support-vector-machines
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar distancia euclidiana, Manhattan, Minkowski y
  coseno.
- Implementar KNN para clasificación y regresión.
- Diagnosticar el parámetro `k` con curvas de validación.
- Estandarizar features antes de KNN (esencial).

## El problema

Tienes un dataset pequeño (cientos de ejemplos) y necesitas
un baseline rápido. KNN te lo da en 5 líneas: "para predecir
un punto nuevo, mira los `k` vecinos más cercanos en el
training set y vota". Sin parámetros que ajustar más allá de
`k` y la métrica. Funciona sorprendentemente bien en datasets
donde la frontera de decisión es local.

## El concepto

**KNN para clasificación.** Para predecir la clase de un punto
nuevo `x`:

1. Calcula `d(x, x_i)` para todo `x_i` en training set.
2. Selecciona los `k` vecinos con menor distancia.
3. Voto mayoritario de las clases de esos `k` vecinos.

**KNN para regresión.** Igual, pero el paso 3 es el promedio
(o promedio ponderado por 1/distancia) de los valores
objetivo de los `k` vecinos.

**Métricas de distancia.** Las más comunes:

- **Euclidiana:** `d(x, y) = sqrt(Σ (x_i - y_i)²)`. La
  *default*. Sensible a outliers.
- **Manhattan (L1):** `d(x, y) = Σ |x_i - y_i|`. Más robusta
  a outliers que L2.
- **Minkowski:** `d(x, y) = (Σ |x_i - y_i|^p)^(1/p)`. L2 es
  `p=2`, L1 es `p=1`.
- **Coseno:** `d(x, y) = 1 - (x·y) / (||x|| ||y||)`. Mide
  ángulo, no magnitud. Útil para texto (TF-IDF).

**Estandarización obligatoria.** Si un feature está en
escala `[0, 1000]` y otro en `[0, 1]`, el primero domina la
distancia. Siempre estandariza con `StandardScaler` (z-score)
o `MinMaxScaler` antes de KNN.

**El parámetro `k`.**

- `k = 1`: frontera de decisión muy irregular, overfitting.
- `k = n`: predice siempre la clase mayoritaria, underfitting.
- Óptimo típicamente entre 3 y 20. Usa validación cruzada.

**Ponderación por distancia.** Variante: el voto de cada
vecino se pondera por `1 / d(x, x_i)`. Los vecinos más
cercanos pesan más, lo que da fronteras más suaves.

**Cuándo usar KNN.**

| Situación | KNN |
|---|---|
| Dataset pequeño (< 10k ejemplos) | Sí, primera opción |
| Baseline rápida | Sí, casi siempre |
| Datos con estructura local | Sí, ideal |
| Alta dimensionalidad (> 100) | Sí, con métrica coseno |
| Millones de ejemplos | No, costo `O(n) por query` |

**Estructuras de aceleración.** Para datasets medianos (10k-
1M), KNN naive `O(n)` por query es prohibitivo. Usa
BallTree o KDTree (sklearn) que reducen el costo a
`O(log n)` en promedio. Para > 1M, considera **ANN**
(Approximate Nearest Neighbors) como FAISS, Annoy o ScaNN,
que sacrifican exactitud por velocidad.

## Constrúyelo

```python
import numpy as np


def euclidiana(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan(a, b):
    return float(np.sum(np.abs(a - b)))


def coseno(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


METRICAS = {"euclidiana": euclidiana, "manhattan": manhattan,
            "coseno": coseno}


def knn(X_train, y_train, X_test, k=3, metrica="euclidiana",
       ponderado=False):
    """KNN classification. ponderado=True: voto por 1/distancia."""
    d_fn = METRICAS[metrica]
    y_pred = []
    for x in X_test:
        dists = np.array([d_fn(x, xt) for xt in X_train])
        idx = np.argsort(dists)[:k]
        vecinos = y_train[idx]
        distancias = dists[idx]
        if ponderado:
            pesos = 1.0 / (distancias + 1e-9)
            clases = np.unique(vecinos)
            scores = np.array([pesos[vecinos == c].sum() for c in clases])
            y_pred.append(clases[np.argmax(scores)])
        else:
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

Eres un asistente que ayuda a tunear KNN. Recibirás la
descripción del dataset y la métrica actual. Tu trabajo:

1. Si accuracy bajo: normalizar features (StandardScaler).
2. Si k=1 sobreajuste: k > 1.
3. Si > 10K muestras: usar BallTree o KDTree.
4. Si features irrelevantes: eliminarlas (afectan distancia).
5. Si métrica euclidiana no funciona: probar Manhattan o
   coseno.
6. Si ponderación por distancia da mejores resultados en
   validación: usarla.
7. Recomendar stratified k-fold para elegir k.
```

## Ejercicios

1. **KNN con pesos**: cada vecino vota con peso `1/distancia`.
2. **KDTree**: usa `sklearn.neighbors.KDTree` para queries
   rápidos en datasets grandes.
3. **Desafío**: implementa KNN con validación leave-one-out
   para encontrar el `k` óptimo.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 2.
- *Nearest Neighbor Methods in Learning and Vision* — Shakhnarovich, Darrell, Indyk.
- scikit-learn KNeighborsClassifier: <https://scikit-learn.org/stable/modules/neighbors.html>.
- FAISS: <https://github.com/facebookresearch/faiss>.

---

> 📚 **Adaptación al español** de la lección "[KNN and Distance Metrics]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
