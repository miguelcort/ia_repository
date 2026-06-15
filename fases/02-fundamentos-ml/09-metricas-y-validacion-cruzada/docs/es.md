# 09 — Métricas y validación cruzada

> La métrica importa más que el modelo. Validar mal es peor que no validar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar accuracy, precision, recall, F1, AUC-ROC.
- Calcular MAE, RMSE, R² para regresión.
- Aplicar k-fold, stratified k-fold y time-series split.
- Diagnosticar data leakage y reporting correcto.

## El problema

Tu modelo tiene 99% de accuracy en el test set. ¿Es bueno? Si
el dataset tiene 99% de ejemplos negativos, el accuracy es
**peor que predecir siempre la clase mayoritaria**. Necesitas
elegir la métrica correcta, validar correctamente, y reportar
intervalos de confianza. La lección cubre lo mínimo
indispensable para no engañar a tu equipo ni a ti mismo.

## El concepto

**Métricas de clasificación binaria.** Para clase positiva vs
negativa, la matriz de confusión es:

|              | Predicho + | Predicho - |
|--------------|------------|------------|
| Real +       | TP         | FN         |
| Real -       | FP         | TN         |

Métricas derivadas:

- **Accuracy:** `(TP + TN) / (TP + FP + TN + FN)`. Engañosa
  con clases desbalanceadas.
- **Precision:** `TP / (TP + FP)`. De los predichos positivos,
  cuántos son realmente positivos.
- **Recall (sensibilidad):** `TP / (TP + FN)`. De los
  realmente positivos, cuántos detectas.
- **F1:** `2 · precision · recall / (precision + recall)`.
  Media armónica, robusta a desbalance.
- **Specificity (TNR):** `TN / (TN + FP)`. Complemento de
  recall.
- **AUC-ROC:** área bajo la curva ROC. Independiente del
  umbral. Para ranking.

**Curva PR vs ROC.** Con clases muy desbalanceadas, la curva
PR (precision-recall) es más informativa que la ROC porque la
ROC puede verse artificialmente bien por el gran número de
TN.

**Métricas de regresión.**

- **MAE (Mean Absolute Error):** `mean(|y - ŷ|)`. Robusta a
  outliers, mismo unidades que `y`.
- **RMSE (Root Mean Squared Error):** `sqrt(mean((y - ŷ)²))`.
  Penaliza más los errores grandes.
- **R² (coeficiente de determinación):** `1 - SSR/SST`. 1 =
  perfecto, 0 = media, negativo = peor que media.
- **MAPE (Mean Absolute Percentage Error):** `mean(|y - ŷ| /
  |y|)`. Interpretable como porcentaje. Problema con `y=0`.

**Validación cruzada.** Para estimar performance de
generalización:

- **Hold-out:** 80% train, 20% test. Simple pero ruidoso.
- **k-fold:** divide en `k` partes, entrena en `k-1`, evalúa
  en la restante. Rota. Promedio.
- **Stratified k-fold:** preserva proporción de clases en
  cada fold. Esencial para clasificación desbalanceada.
- **Time-series split:** split cronológico, sin shuffle. Para
  series de tiempo.
- **Leave-one-out:** `k = n`. Costoso pero bajo sesgo.
- **Nested CV:** CV interno para hiperparámetros, CV externo
  para evaluación.

**Data leakage.** La causa #1 de modelos que fallan en
producción. Ocurre cuando información del test set o del
futuro contamina el training. Ejemplos:

- Escalar con todo el dataset antes de split.
- Imputar valores faltantes con la media global antes de
  split.
- Usar `fit_transform` en lugar de `fit` + `transform`.
- Time-series split con shuffle.

**Reporting correcto.** Un paper de ML responsable reporta:

1. Métrica principal con intervalo de confianza (vía
   bootstrap o CV).
2. Comparación contra un baseline (random, majority class,
   modelo anterior).
3. Tamaño del test set y método de split.
4. Hiperparámetros seleccionados y método (grid, random,
   Bayesian).
5. Recursos computacionales (GPU, tiempo, costo).

## Constrúyelo

```python
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, positivo=1):
    tp = np.sum((y_pred == positivo) & (y_true == positivo))
    fp = np.sum((y_pred == positivo) & (y_true != positivo))
    fn = np.sum((y_pred != positivo) & (y_true == positivo))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def confusion_matrix(y_true, y_pred, clases=None):
    if clases is None:
        clases = np.unique(np.concatenate([y_true, y_pred]))
    K = len(clases)
    M = np.zeros((K, K), dtype=int)
    for i, c_true in enumerate(clases):
        for j, c_pred in enumerate(clases):
            M[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
    return M


def auc_roc(y_true, scores):
    """AUC-ROC via Mann-Whitney U."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Comparaciones: ¿cuántas veces un positivo > un negativo?
    count = 0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(count / (n_pos * n_neg))


def k_fold_indices(n, k=5, semilla=0):
    """Genera índices de k-fold split."""
    rng = np.random.default_rng(semilla)
    indices = rng.permutation(n)
    return np.array_split(indices, k)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-metricas-elegir
fase: 02
leccion: 09
---

Eres un asistente que ayuda a elegir y reportar métricas. Reci-
birás el problema (clasificación/regresión, balance de clases,
costo de errores FP/FN, contexto de aplicación). Tu trabajo:

1. Clasificación balanceada: accuracy, F1, AUC-ROC.
2. Clasificación desbalanceada: F1, AUC-PR, precision-recall.
3. Si FP y FN tienen costos diferentes: definir una
   métrica ponderada o ajustar umbral.
4. Regresión con outliers: MAE. Sin outliers: RMSE.
5. Regresión con R² fácil de interpretar: R².
6. Siempre reportar IC (bootstrap) o std sobre k-fold.
7. Comparar contra un baseline simple (mayority class,
   media, último valor).
8. Si time-series: time-series split, nunca shuffle.
```

## Ejercicios

1. **Métricas desde cero**: implementa F1, AUC-ROC, y
   confusion matrix sin sklearn.
2. **K-fold stratified**: implementa k-fold que preserve
   proporción de clases en cada fold.
3. **Desafío**: implementa nested cross-validation para
   seleccionar hiperparámetros sin leakage.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 5.
- *Pattern Recognition and Machine Learning* — Bishop.
- scikit-learn model_evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>.
- *Deep Learning* — Goodfellow, Bengio, Courville (cap. 8).

---

> 📚 **Adaptación al español** de la lección "[Metrics and Cross-Validation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
