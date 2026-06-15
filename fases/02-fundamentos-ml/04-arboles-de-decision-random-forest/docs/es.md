# 04 — Árboles de decisión y Random Forest

> Los árboles son interpretables. Los Random Forest son la versión poderosa: ensemble de árboles que se equivocan en lugares distintos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar el algoritmo ID3/CART para árboles de decisión.
- Calcular Gini y entropía como criterios de split.
- Diagnosticar overfitting en árboles y aplicar poda.
- Construir un Random Forest como ensemble de árboles
  bootstrap.

## El problema

Necesitas un modelo que te diga qué clientes van a *churn*
y, además, que el equipo de marketing pueda **entender** por
qué. Una red neuronal te da accuracy alta pero su
interpretabilidad es opaca. Un árbol de decisión te da un
diagrama que puedes poner en una slide. Y un Random Forest te
da el poder predictivo de un ensemble sin perder
interpretabilidad via *feature importance*.

## El concepto

**Árbol de decisión.** Estructura jerárquica de decisiones
binarias sobre features. Cada nodo interno pregunta "¿feature
`x_j` <= umbral?" y divide el dataset. Las hojas contienen
predicciones. Para clasificación, voto mayoritario en la hoja;
para regresión, media.

**Algoritmo CART.** Para cada nodo:

1. Para cada feature `x_j` y cada umbral candidato `t`:
   calcula la impureza ponderada del split.
2. Elige el split que minimiza la impureza ponderada.
3. Repite recursivamente en los hijos hasta criterio de
   parada (profundidad máxima, muestras mínimas por hoja,
   mejora mínima).

**Criterios de impureza.**

- **Gini:** `G = 1 - Σ p_k²`. Mide qué tan mezcladas están
  las clases en el nodo. Gini = 0 cuando todos los ejemplos
  son de la misma clase. Computacionalmente más barato.
- **Entropía:** `H = -Σ p_k log₂(p_k)`. Tiene base teórica en
  teoría de información (bits necesarios para codificar la
  clase). Gini y entropía dan resultados similares en la
  práctica.

**Overfitting y poda.** Un árbol sin restricciones puede
memorizar el training set. Soluciones:

- Limitar profundidad máxima (`max_depth`).
- Mínimo de muestras por hoja (`min_samples_leaf`).
- Mínimo de mejora de impureza para split (`min_impurity_decrease`).
- Poda post-hoc: construir árbol completo y remover nodos
  cuya eliminación no degrada validation.

**Feature importance.** Para cada feature, suma la reducción
de impureza en todos los splits que la usan. Da una
importancia relativa de cada feature, útil para
interpretabilidad y feature selection.

**Random Forest.** Ensemble de árboles donde cada árbol se
entrena con un subconjunto bootstrap de los datos y solo
considera un subconjunto aleatorio de features en cada split.
El voto mayoritario (clasificación) o promedio (regresión) de
los árboles da la predicción final. Reduce la varianza de un
solo árbol sin aumentar el sesgo.

**Hiperparámetros clave.**

| Hiperparámetro | Default sklearn | Efecto |
|---|---|---|
| `n_estimators` | 100 | Más árboles = más estable, más lento |
| `max_depth` | None | Limita overfitting |
| `min_samples_split` | 2 | Mínimo para splitear un nodo |
| `min_samples_leaf` | 1 | Mínimo en hojas |
| `max_features` | sqrt(n) | Features por split (auto para clasif) |
| `bootstrap` | True | Si False, usa todo el dataset |

**Cuándo usar árboles / Random Forest.**

| Situación | Recomendación |
|---|---|
| Datos tabulares, pequeña escala | Árbol o RF |
| Interpretabilidad | Árbol pequeño |
| Accuracy sin importar interpretabilidad | RF o XGBoost |
| Datos no tabulares (imagen, texto, audio) | Mejor una red neuronal |

## Constrúyelo

```python
import numpy as np


def gini(y):
    """Impureza de Gini: 0 si puro, 0.5 si 50/50 en binario."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def entropia(y):
    """Entropía de Shannon: bits para codificar la clase."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mejor_split(X, y, criterio="gini"):
    """Encuentra el mejor split (feature, umbral) según Gini o entropía."""
    n, d = X.shape
    impurity_fn = gini if criterio == "gini" else entropia
    base_impurity = impurity_fn(y)
    mejor = {"ganancia": -np.inf, "feature": None, "umbral": None}
    for j in range(d):
        umbrales = np.unique(X[:, j])
        for t in umbrales:
            izq = y[X[:, j] <= t]
            der = y[X[:, j] > t]
            if len(izq) == 0 or len(der) == 0:
                continue
            weighted = (len(izq) * impurity_fn(izq) +
                        len(der) * impurity_fn(der)) / n
            ganancia = base_impurity - weighted
            if ganancia > mejor["ganancia"]:
                mejor.update({"ganancia": ganancia,
                              "feature": j, "umbral": t})
    return mejor
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tree-rf
fase: 02
leccion: 04
---

Eres un asistente que ayuda a elegir entre árbol de decisión y
Random Forest. Recibirás el tamaño del dataset, el número de
features, y el requerimiento de interpretabilidad. Tu trabajo:

1. Si interpretabilidad es clave y accuracy modesta sirve:
   árbol de decisión pequeño.
2. Si accuracy es lo primero: Random Forest con
   `n_estimators=100+` y `max_features=sqrt(n)`.
3. Si el dataset es muy grande o tienes features no tabulares:
   XGBoost / LightGBM.
4. Advertir contra árboles sin poda: overfit seguro.
5. Sugerir `class_weight=balanced` para clases desbalanceadas.
6. Reportar feature importance después del entrenamiento.
```

## Ejercicios

1. **Árbol completo**: implementa la estructura recursiva del
   árbol y predice en un dataset de juguete.
2. **Random Forest**: bootstrap + feature subsampling +
   promedio de predicciones.
3. **Desafío**: implementa poda por reduced error pruning
   (post-hoc).

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 8.
- *Elements of Statistical Learning* — cap. 15.
- scikit-learn DecisionTreeClassifier: <https://scikit-learn.org/stable/modules/tree.html>.
- scikit-learn RandomForestClassifier: <https://scikit-learn.org/stable/modules/ensemble.html#random-forests>.

---

> 📚 **Adaptación al español** de la lección "[Decision Trees and Random Forest]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
