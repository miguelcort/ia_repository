# 03 — Regresión logística y clasificación

> Regresión logística es el "hello world" de clasificación. Simple, interpretable, y base de redes neuronales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-regresion-lineal-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la función sigmoide y la regresión logística
  desde cero con NumPy.
- Ajustar los pesos por descenso de gradiente con
  cross-entropy.
- Diagnosticar accuracy, matriz de confusión, precision,
  recall y F1.
- Extender a clasificación multiclase con softmax +
  cross-entropy categórica.

## El problema

Tienes datos de clientes (edad, ingreso, historial) y quieres
predecir si van a *default* en su préstamo. Es un problema de
clasificación binaria: la variable objetivo es 0 o 1. La
regresión lineal no funciona aquí porque puede predecir valores
fuera de `[0, 1]`. Necesitas un modelo cuya salida sea una
**probabilidad** — eso es regresión logística.

## El concepto

**Función sigmoide.** Mapea cualquier número real a
`(0, 1)`:

```text
σ(z) = 1 / (1 + exp(-z))
```

Tiene derivada conveniente: `σ'(z) = σ(z) (1 - σ(z))`. Es la
base de la regresión logística y, por extensión, de las
neuronas artificiales con activación sigmoide (las primeras
redes neuronales).

**Modelo.** Para features `x ∈ R^d`:

```text
p(y=1 | x; w) = σ(w^T x + b)
```

Aprendes `w` y `b` maximizando la log-verosimilitud
(equivalente a minimizar la cross-entropy binaria).

**Cross-entropy binaria (pérdida).** Para un ejemplo `(x, y)`:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

Es la divergencia KL entre la distribución verdadera `Bernoulli(y)`
y la predicha `Bernoulli(p)`. Es **convexa** en `w` (buen
objetivo para optimización) y su gradiente tiene forma
particularmente simple: `∂L/∂w = (p - y) x`.

**Algoritmo de entrenamiento.** Inicializa `w = 0`, itera:

```text
p = σ(X w + b)
grad = X^T (p - y) / n
w ← w - lr · grad
```

Converge en `O(n_iter)` para learning rate razonable.
Alternativa: usar L-BFGS (scipy.optimize.minimize) para
convergencia más rápida.

**Multiclase con softmax.** Para `K > 2` clases, reemplaza
sigmoide con softmax y cross-entropy binaria con cross-entropy
categórica. La salida del modelo es un vector de `K`
probabilidades que suman 1.

**Métricas de clasificación.** Accuracy no es suficiente
cuando las clases están desbalanceadas. Complementa con:

- **Matriz de confusión:** `K x K` con TP, FP, FN, TN.
- **Precision:** de los predichos positivos, cuántos son
  realmente positivos.
- **Recall (sensibilidad):** de los realmente positivos,
  cuántos detectas.
- **F1:** media armónica de precision y recall.
- **AUC-ROC:** área bajo la curva ROC, independiente del
  umbral.

**Cuándo usar regresión logística.**

| Situación | Regresión logística |
|---|---|
| Features tabulares, lineales | Sí, primera opción |
| Alta interpretabilidad | Sí, coeficientes interpretables |
| Baseline rápida | Sí, casi siempre el baseline |
| Datos no lineales | Con features polinomiales o kernel |
| Texto con TF-IDF | Sí, sorprendentemente efectivo |

## Constrúyelo

```python
import numpy as np


def sigmoid(x):
    """σ(x) = 1 / (1 + exp(-x)). Estable numéricamente para x grandes."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def ajustar(X, y, lr=0.1, epochs=1000, tol=1e-6):
    """Ajusta w por descenso de gradiente con cross-entropy."""
    n, d = X.shape
    w = np.zeros(d + 1)
    X_aug = np.column_stack([np.ones(n), X])
    for i in range(epochs):
        p = sigmoid(X_aug @ w)
        grad = X_aug.T @ (p - y) / n
        w -= lr * grad
        if np.linalg.norm(grad, 1) < tol:
            break
    return w


def predecir(X, w, umbral=0.5):
    """Predice clase 0/1 con umbral configurable."""
    n = X.shape[0]
    X_aug = np.column_stack([np.ones(n), X])
    return (sigmoid(X_aug @ w) >= umbral).astype(int)


def softmax(X, w):
    """Multiclase: K probabilidades que suman 1."""
    logits = X @ w
    logits -= logits.max(axis=1, keepdims=True)  # estabilidad
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)
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

Eres un asistente que ayuda a diagnosticar problemas de
clasificación binaria. Recibirás el accuracy, matriz de
confusión y descripción del problema. Tu trabajo:

1. Si accuracy ~50% y clases balanceadas: el modelo no
   está aprendiendo, considera features o modelo.
2. Si overfitting en train pero no en test: regularizar.
3. Si clases desbalanceadas: usar class_weight o
   resampling (SMOTE).
4. Si precision alta pero recall bajo: bajar umbral.
5. Si recall alto pero precision bajo: subir umbral.
6. Si accuracy estable pero F1 bajo: usar F1 como
   objetivo y ajustar umbral.
```

## Ejercicios

1. **Multiclase**: implementa softmax + cross-entropy para
   `3+` clases.
2. **Regularización L2**: agrega `λ||w||²` al gradiente y
   observa el efecto en overfitting.
3. **Desafío**: implementa regresión logística regularizada
   con *early stopping* y compara con la versión sin
   *early stopping*.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — James, Witten,
  Hastie, Tibshirani (cap. 4).
- scikit-learn LogisticRegression: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>.
- *Pattern Recognition and Machine Learning* — Bishop (cap. 4).

---

> 📚 **Adaptación al español** de la lección "[Logistic Regression and Classification]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
