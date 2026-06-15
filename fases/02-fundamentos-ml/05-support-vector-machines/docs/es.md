# 05 — Support Vector Machines (SVM)

> SVM encuentra el hiperplano que maximiza el margen. Con kernels, resuelve problemas no lineales. Sigue siendo fuerte en datasets pequeños/medianos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Derivar la formulación primal y dual de SVM lineal.
- Resolver el problema dual con QP o SMO.
- Aplicar el *kernel trick* para problemas no lineales.
- Diagnosticar el papel de `C` y `gamma`.

## El problema

Tienes 1000 ejemplos y 50 features. La regresión logística
funciona pero quieres el **mejor clasificador lineal** en
sentido geométrico: el hiperplano que maximiza la distancia a
los puntos más cercanos. SVM es ese clasificador. Con
*kernels*, extiende la idea a separadores no lineales sin
pagar el costo computacional de transformar los features.

## El concepto

**Formulación primal.** Dados datos `(x_i, y_i)` con
`y_i ∈ {-1, +1}`, SVM lineal busca `w, b` que minimicen:

```text
(1/2) ||w||² + C · Σ ξ_i
```

sujeto a `y_i (w·x_i + b) ≥ 1 - ξ_i` y `ξ_i ≥ 0`. Los `ξ_i`
son *slack variables* que permiten violaciones; `C` controla
el *trade-off* entre margen y errores.

**Interpretación geométrica.** El término `(1/2) ||w||²`
minimiza la norma de los pesos, lo que es equivalente a
**maximizar el margen** (distancia entre el hiperplano y los
*puntos más cercanos*). El hiperplano SVM está en el centro
del "callejón" más ancho que separa las clases.

**Formulación dual.** Aplicando Lagrange, el problema es
equivalente a:

```text
max Σ α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j (x_i · x_j)
sujeto a 0 ≤ α_i ≤ C, Σ α_i y_i = 0
```

Los `α_i` son los multiplicadores de Lagrange. Los puntos con
`α_i > 0` son los **support vectors** — los únicos que
definen el hiperplano.

**Kernels.** El dual solo requiere productos punto `x_i · x_j`.
Si reemplazas este producto por un kernel `K(x_i, x_j)`, SVM
opera implícitamente en un espacio de features de dimensión
infinita sin pagar el costo de la transformación. Los kernels
clásicos:

- **Lineal:** `K(x, y) = x·y`. Sin transformación.
- **RBF (gaussiano):** `K(x, y) = exp(-γ ||x - y||²)`. El más
  usado. Convierte a un clasificador de radio.
- **Polinomial:** `K(x, y) = (γ x·y + r)^d`. Para relaciones
  polinomiales.

**Hiperparámetros.**

| Parámetro | Efecto |
|---|---|
| `C` | Alto C = más estricto (menos margen, menos error). Bajo C = margen ancho, más tolerante. |
| `gamma` (RBF) | Alto gamma = cada punto tiene influencia muy local (riesgo de overfitting). Bajo gamma = influencia más amplia. |
| `kernel` | Lineal, RBF, polinomial, sigmoid. |

**Cuándo usar SVM.**

| Situación | Recomendación |
|---|---|
| Datos tabulares pequeños/medianos | SVM RBF |
| Muchas features, pocos datos (texto) | SVM lineal |
| Datos no tabulares (imagen) | Mejor CNN |
| Necesidad de probabilidad | Calibrar con Platt scaling |
| Dataset > 100k | Usar SGDClassifier o LinearSVC |

## Constrúyelo

```python
import numpy as np
from scipy.optimize import minimize


def svm_lineal(X, y, C=1.0, n_iter=1000, lr=0.01):
    """SVM lineal primal por descenso de gradiente subgradiente."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        for i in range(n):
            margin = y[i] * (X[i] @ w + b)
            if margin >= 1:
                w -= lr * w
            else:
                w -= lr * (w - C * y[i] * X[i])
                b += lr * C * y[i]
    return w, b


def kernel_rbf(X, Y, gamma=0.5):
    """K(x, y) = exp(-gamma ||x - y||^2)."""
    XX = (X ** 2).sum(axis=1, keepdims=True)
    YY = (Y ** 2).sum(axis=1, keepdims=True)
    dist = XX + YY.T - 2 * X @ Y.T
    return np.exp(-gamma * dist)


def dual_svm(X, y, C=1.0, kernel=kernel_rbf, gamma=0.5):
    """Resuelve el SVM dual con scipy.optimize.
    Variables: alpha_i en [0, C]. Restricción: Σ alpha_i y_i = 0."""
    K = kernel(X, X, gamma=gamma) if gamma is not None else kernel(X, X)

    def neg_dual(alpha):
        return -np.sum(alpha) + 0.5 * np.sum(
            (alpha * y)[:, None] * (alpha * y)[None, :] * K
        )

    def neg_dual_grad(alpha):
        return -np.ones_like(alpha) + (alpha * y) @ (K * y[:, None])

    n = len(y)
    constraints = {"type": "eq", "fun": lambda a: np.sum(a * y)}
    bounds = [(0, C)] * n
    res = minimize(
        neg_dual, np.zeros(n), jac=neg_dual_grad,
        bounds=bounds, constraints=constraints, method="SLSQP"
    )
    return res.x
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-svm-elegir
fase: 02
leccion: 05
---

Eres un asistente que ayuda a configurar SVM. Recibirás la
descripción del problema (número de ejemplos, dimensionalidad,
tipo de features, requerimiento de accuracy). Tu trabajo:

1. Si n < 10k y d < 1000: SVM RBF.
2. Si texto (n grande, d muy grande): SVM lineal.
3. Si imagen: mejor CNN.
4. Advertir: SVM escala O(n²) a O(n³) — datasets > 100k son
   prohibitivos sin SGD o LinearSVC.
5. Sugerir GridSearch sobre C ∈ {0.1, 1, 10, 100} y
   gamma ∈ {0.001, 0.01, 0.1, 1}.
6. Usar stratified k-fold cross-validation.
7. Reportar classification report (precision, recall, F1).
```

## Ejercicios

1. **SVM lineal desde cero**: implementa la versión primal
   con subgradiente y compara con `sklearn.svm.LinearSVC`.
2. **Kernel RBF**: aplica SVM con kernel RBF a un dataset
   con frontera circular y verifica la separación.
3. **Desafío**: implementa SMO (Sequential Minimal
   Optimization) para resolver el dual.

## Lecturas recomendadas

- *An Introduction to Statistical Learning* — cap. 9.
- *Learning with Kernels* — Schölkopf & Smola.
- scikit-learn svm: <https://scikit-learn.org/stable/modules/svm.html>.
- *Pattern Recognition and Machine Learning* — Bishop, cap. 7.

---

> 📚 **Adaptación al español** de la lección "[Support Vector Machines]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
