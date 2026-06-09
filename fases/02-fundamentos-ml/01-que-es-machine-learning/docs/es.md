# Que es Machine Learning

> Machine learning es hacer que las maquines aprendan de los datos en vez de programar reglas.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 00-configuracion-y-herramientas, 01-fundamentos-matematicas
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Generar datos sinteticos lineales.
- Implementar MSE y division train/test.
- Diagnosticar overfitting vs underfitting.

## Constrúyelo

```python
import numpy as np


def generar_datos_lineales(n=100, ruido=0.1, semilla=0):
    rng = np.random.default_rng(semilla)
    X = np.linspace(0, 5, n)
    y = 2 * X + 1 + ruido * rng.normal(size=n)
    return X, y


def error_cuadratico_medio(y_v, y_p):
    return float(np.mean((y_v - y_p) ** 2))


def dividir_train_test(X, y, prop_train=0.8, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(X)
    indices = rng.permutation(n)
    n_train = int(n * prop_train)
    return X[indices[:n_train]], y[indices[:n_train]], X[indices[n_train:]], y[indices[n_train:]]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ml-elegir
fase: 02
leccion: 01
---

1. Con etiquetas: supervisado.
2. Sin etiquetas: no supervisado.
3. Con recompensa: RL.
4. <10K: modelos simples.
5. >100K: deep learning.
```

## Ejercicios

1. **Train/val/test split**: implementa division 60/20/20.
2. **Bias-variance**: implementa una funcion que mide ambos en
   funcion del tamano del train set.
3. **Desafio**: implementa K-fold cross-validation.

## Lecturas recomendadas

- "An Introduction to Statistical Learning" (James, Witten, et al.): <https://www.statlearning.com/>
- scikit-learn: <https://scikit-learn.org/>

---

> 📚 **Adaptación al español** de la lección "[What is Machine Learning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).