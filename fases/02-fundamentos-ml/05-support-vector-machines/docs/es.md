# Support Vector Machines (SVM)

> SVM fue el estado del arte antes de deep learning. Hoy sigue siendo fuerte en datos pocos y alta dimensionalidad.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-regresion-logistica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar kernels (lineal, RBF).
- Calcular margen geometrico y hinge loss.
- Diagnosticar el parametro C.

## Constrúyelo

```python
import numpy as np


def kernel_rbf(x, y, gamma=1.0):
    diff = np.asarray(x) - np.asarray(y)
    return float(np.exp(-gamma * np.dot(diff, diff)))


def margen(X, y, w):
    return float(np.min(y * (X @ w) / np.linalg.norm(w)))


def hinge_loss(y, scores, reg=1.0):
    margins = np.maximum(0, 1 - y * scores)
    return float(np.mean(margins) + 0.5 * reg * np.dot(scores, scores))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-svm-tuning
fase: 02
leccion: 05
---

1. Sobreajuste: C menor.
2. Subajuste: C mayor o kernel flexible.
3. Grid search C x gamma.
4. >100K: LinearSVC.
```

## Ejercicios

1. **SVM primal con SGD**: implementa el entrenamiento por GD.
2. **SVM con kernel polinomial**: implementa y compara con RBF.
3. **Desafio**: implementa SVR (regresion con epsilon-insensitive loss).

## Lecturas recomendadas

- "Understanding Machine Learning" cap. 17 (Shalev-Shwartz)
- scikit-learn SVC: <https://scikit-learn.org/stable/modules/svm.html>

---

> 📚 **Adaptación al español** de la lección "[Support Vector Machines]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).