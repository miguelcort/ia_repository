# 11 — Métodos de ensemble: boosting, bagging, stacking

> Ningún modelo individual es perfecto. Los ensembles combinan varios para que se equivoquen en lugares distintos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-arboles-de-decision-random-forest
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar bagging y Random Forest desde cero.
- Implementar AdaBoost y Gradient Boosting.
- Comparar voting, averaging, stacking y blending.
- Diagnosticar cuándo un ensemble mejora sobre un solo
  modelo.

## El problema

Tienes un modelo que se equivoca en 30% de los casos.
Entrenas otro modelo que se equivoca en 30% también, pero en
casos distintos. Si combinas las predicciones, el error
baja. Esa es la promesa del ensembling: múltiples modelos
**diversos** (no solo múltiples copias) son mejores que uno
solo. La lección cubre las tres familias: bagging, boosting y
stacking.

## El concepto

**Bagging (Bootstrap AGGregatING).** Entrena `B` modelos con
`B` muestras bootstrap (muestreo con reemplazo) del dataset
original. Promedia las predicciones (regresión) o vota
(clasificación). Reduce la **varianza** del modelo sin
aumentar el sesgo. Random Forest es la aplicación más famosa:
árboles con feature subsampling adicional.

**Boosting.** Entrena modelos secuencialmente, donde cada
modelo nuevo corrige los errores del anterior. Reduce el
**sesgo** (los modelos simples se vuelven complejos al
combinarse). Dos familias principales:

- **AdaBoost:** ajusta pesos de los ejemplos; los ejemplos
  mal clasificados ganan peso, los bien clasificados pierden.
  El modelo nuevo se enfoca en los difíciles. Al final,
  predice con voto ponderado.
- **Gradient Boosting:** ajusta el modelo nuevo al **gradiente
  negativo** de la pérdida con respecto a la predicción del
  modelo actual. Es la formalización moderna que da lugar a
  XGBoost, LightGBM y CatBoost.

**Stacking.** Entrena varios modelos base (diversos:
Random Forest, SVM, red neuronal) y luego un **meta-learner**
que toma las predicciones de los modelos base como input.
El meta-learner aprende cuándo confiar en cada modelo base.
Es el más poderoso pero también el más propenso a overfitting.

**Blending.** Variante de stacking donde el meta-learner se
entrena con un hold-out separado, no con k-fold. Menos
overfitting pero menos datos para entrenar.

**Voting vs averaging.** Para ensembles de modelos que
producen probabilidades:

- **Hard voting:** voto mayoritario de las clases predichas.
- **Soft voting:** promedio de las probabilidades.
- Soft voting suele ser mejor cuando los modelos están bien
  calibrados.

**Cuándo usar cada técnica.**

| Técnica | Cuándo |
|---|---|
| Bagging (RF) | Modelo con alta varianza, datos tabulares |
| Boosting (XGBoost) | Modelo con alto sesgo, datos tabulares, quieres el mejor accuracy |
| Stacking | Tienes varios modelos buenos y quieres exprimir más |
| Voting | Tienes varios modelos razonables, quieres un ensemble simple |

**Hiperparámetros clave de Gradient Boosting.**

- `n_estimators`: número de árboles. Más = más complejo.
- `learning_rate`: contribución de cada árbol. Bajo (0.01-
  0.1) requiere más árboles pero generaliza mejor.
- `max_depth`: profundidad de cada árbol. Típico 3-8.
- `subsample`: fracción de muestras por árbol (Stochastic GB).

**Trampas.**

- **Bagging con modelos idénticos:** no aporta nada.
  Diversidad es clave.
- **Boosting con modelos sobreajustados:** diverge. Usa
  shrinkage (learning_rate bajo) y regularización.
- **Stacking con data leakage:** el meta-learner puede
  overfittear si ve las predicciones de modelos que ya vieron
  los datos. Usa k-fold OOF (out-of-fold) para el meta-learner.

## Constrúyelo

```python
import numpy as np
from collections import Counter


def bagging(modelo_class, X, y, n_estimators=10, muestra_frac=0.8,
            semilla=0):
    """Bagging de cualquier clasificador con .fit() y .predict()."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    modelos = []
    for _ in range(n_estimators):
        idx = rng.choice(n, size=int(n * muestra_frac), replace=True)
        m = modelo_class()
        m.fit(X[idx], y[idx])
        modelos.append(m)
    return modelos


def voting_ensemble(modelos, X, tipo="soft"):
    """Combina predicciones de un ensemble de modelos."""
    if tipo == "soft":
        # Promedio de probabilidades (asumimos predict_proba)
        probs = np.mean([m.predict_proba(X) for m in modelos], axis=0)
        return np.argmax(probs, axis=1)
    else:
        # Voto mayoritario
        preds = np.array([m.predict(X) for m in modelos])
        return np.array([Counter(preds[:, i]).most_common(1)[0][0]
                        for i in range(preds.shape[1])])


class GradientBoostingRegressor:
    """GB regresión con árboles de decisión de profundidad 1."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = 0.0

    def _fit_tree(self, X, residuals):
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X, residuals)
        return tree

    def fit(self, X, y):
        self.init_pred = float(np.mean(y))
        pred = np.full(len(y), self.init_pred)
        for _ in range(self.n_estimators):
            residuals = y - pred
            tree = self._fit_tree(X, residuals)
            self.trees.append(tree)
            pred += self.lr * tree.predict(X)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ensemble-elegir
fase: 02
leccion: 11
---

Eres un asistente que ayuda a elegir técnicas de ensemble.
Recibirás el dataset, el modelo base, y la métrica actual.
Tu trabajo:

1. Si modelo base tiene alta varianza: bagging (Random
   Forest).
2. Si modelo base tiene alto sesgo: boosting (XGBoost,
   LightGBM).
3. Si tienes modelos muy diferentes: stacking con
   meta-learner.
4. Si quieres ensemble simple y robusto: voting (soft
   si tienes probabilidades, hard si no).
5. Advertir: bagging sin diversidad no aporta.
6. Advertir: boosting con learning_rate alto diverge.
7. Sugerir `n_estimators=100+` con `learning_rate=0.05-0.1`
   para GB.
8. Validar el ensemble con k-fold CV.
```

## Ejercicios

1. **AdaBoost desde cero**: implementa AdaBoost con árboles
   de decisión de profundidad 1 (stumps).
2. **Stacking con OOF**: implementa stacking donde las
   predicciones de los modelos base se generan con
   out-of-fold para evitar leakage.
3. **Desafío**: implementa Gradient Boosting desde cero
   sin sklearn, usando mínimos cuadrados en cada iteración.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 8.
- *Elements of Statistical Learning* — cap. 10, 15, 16.
- XGBoost paper: Chen & Guestrin, 2016.
- LightGBM paper: Ke et al., 2017.
- scikit-learn ensemble: <https://scikit-learn.org/stable/modules/ensemble.html>.

---

> 📚 **Adaptación al español** de la lección "[Ensemble Methods]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
