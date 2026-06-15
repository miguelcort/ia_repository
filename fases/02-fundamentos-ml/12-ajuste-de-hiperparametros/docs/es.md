# 12 — Ajuste de hiperparámetros

> El modelo correcto con hiperparámetros equivocados es peor que un modelo mediocre con hiperparámetros tuneados. Grid y Bayesian search son obligatorios.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-metricas-y-validacion-cruzada
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar grid search, random search y Bayesian
  optimization.
- Diagnosticar cuándo cada técnica es apropiada según el
  espacio de búsqueda.
- Aplicar Optuna para optimización con *pruning* temprano.
- Reportar hiperparámetros óptimos con intervalos de
  confianza.

## El problema

Tienes un XGBoost con 8 hiperparámetros. Si los probaras todos
en una grilla de 5 valores cada uno, serían `5^8 = 390,625`
combinaciones. Demasiado. Si solo pruebes 100 al azar, puedes
perderte el óptimo. Si haces 1000 iteraciones de Bayesian
search con un prior inteligente, llegas cerca del óptimo
con `O(iterations)`. La lección cubre las tres técnicas y
cuándo usar cada una.

## El concepto

**Grid search.** Prueba todas las combinaciones en una grilla
discretizada. Exhaustivo pero caro. Funciona para espacios
de búsqueda pequeños (≤ 4 hiperparámetros).

```python
for lr in [0.01, 0.1, 1.0]:
    for depth in [3, 5, 7]:
        for n_est in [100, 300, 1000]:
            score = cross_val_score(model, X, y, params)
```

**Random search.** Muestrea combinaciones al azar del espacio
de búsqueda. Sorprendentemente efectivo, especialmente cuando
algunos hiperparámetros importan más que otros. Funciona bien
para espacios de 4-10 hiperparámetros.

**Bayesian optimization.** Construye un modelo surrogate
(típicamente un GP o un TPE — Tree-structured Parzen
Estimator) que predice la performance del modelo en cada
punto del espacio. En cada iteración:

1. Usa el surrogate para elegir el siguiente punto (maximizando
   un *acquisition function* como Expected Improvement).
2. Evalúa el modelo en ese punto.
3. Actualiza el surrogate.

Es `O(iterations)` en vez de `O(grid_size)` y maneja mejor
espacios de búsqueda grandes o continuos. Optuna implementa
TPE y es la opción moderna dominante.

**Hyperband y pruning.** Para modelos costosos (deep
learning), no quieres entrenar un modelo por 100 epochs para
descubrir que es malo. Hyperband entrena muchas
configuraciones con pocos epochs y *prunea* las malas. Optuna
integra pruning con `MedianPruner`.

**Nested cross-validation.** Para evitar sobreajustar los
hiperparámetros al set de validación:

- CV externo: divide train/test.
- CV interno: dentro de train, divide train/val y busca
  hiperparámetros.

Si tu CV interno reporta `0.92 ± 0.01` y tu CV externo reporta
`0.85 ± 0.03`, el espacio de búsqueda es demasiado pequeño o
los datos son pocos. Ambos números deben concordar.

**Cuándo usar cada técnica.**

| Técnica | Espacio de búsqueda | Recursos | Uso típico |
|---|---|---|---|
| Grid | < 100 combinaciones | Cualquiera | Baseline, espacios pequeños |
| Random | < 10 hiperparámetros | Moderados | Primer paso rápido |
| Bayesian (Optuna) | 5-50 hiperparámetros | Moderados | Producción |
| Hyperband | 10+ hiperparámetros, modelo costoso | Altamente paralelos | Deep learning |

**Trampas.**

- **Optimizar en test set:** siempre optimiza en validación,
  mide en test.
- **No reportar varianza:** "XGBoost tiene 0.92" no dice
  nada; "0.92 ± 0.015 sobre 5-fold" sí.
- **Espacio de búsqueda mal definido:** incluir rangos
  razonables. `learning_rate` ∈ [1e-5, 1e-1], no [0, 1000].
- **No usar *pruning*:** con modelos costosos, es
  prohibitivo entrenar todos a *full epochs*.

## Constrúyelo

```python
import numpy as np
from itertools import product


def grid_search(model_fn, params_grid, X, y, cv=5, scoring="accuracy"):
    """Grid search exhaustivo con k-fold cross-validation."""
    from sklearn.model_selection import cross_val_score
    resultados = []
    for params in product(*params_grid.values()):
        keys = list(params_grid.keys())
        kw = dict(zip(keys, params))
        modelo = model_fn(**kw)
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        resultados.append({"params": kw, "mean": float(scores.mean()),
                            "std": float(scores.std())})
    resultados.sort(key=lambda r: r["mean"], reverse=True)
    return resultados


def random_search(model_fn, params_dist, X, y, n_iter=20, cv=5,
                  scoring="accuracy", semilla=0):
    """Random search muestreando de distribuciones."""
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(semilla)
    resultados = []
    for _ in range(n_iter):
        params = {k: v(rng) for k, v in params_dist.items()}
        modelo = model_fn(**params)
        scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
        resultados.append({"params": params, "mean": float(scores.mean()),
                            "std": float(scores.std())})
    resultados.sort(key=lambda r: r["mean"], reverse=True)
    return resultados
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-hyperparam-tuning
fase: 02
leccion: 12
---

Eres un asistente que ayuda a configurar la búsqueda de
hiperparámetros. Recibirás el modelo, el espacio de búsqueda, y
los recursos disponibles. Tu trabajo:

1. Si espacio < 100 combinaciones: grid search exhaustivo.
2. Si espacio mayor: random search con 50-100 iteraciones
   como baseline.
3. Si quieres exprimir el último 1% de accuracy: Optuna
   (Bayesian) con 100-500 trials.
4. Si el modelo es costoso (deep learning): hyperband o
   Optuna con `MedianPruner` para podar trials malos
   temprano.
5. Siempre: nested cross-validation o split train/val/test
   para reportar resultados honestos.
6. Reportar `mean ± std` sobre los folds, no solo el mejor.
7. Logging: usar MLflow o W&B para trackear todos los trials.
```

## Ejercicios

1. **Optuna con pruning**: implementa un estudio Optuna con
   `MedianPruner` para una red neuronal pequeña.
2. **Hyperband**: implementa Hyperband desde cero con
   bracket de brackets sucesivos.
3. **Desafío**: implementa Bayesian optimization con un
   Gaussian Process surrogate desde cero.

## Lecturas recomendadas

- *Hyperparameter Optimization* — book by Theyser, 2021.
- *Algorithms for Hyper-Parameter Optimization* — Bergstra,
  Bengio, 2011.
- Optuna: <https://optuna.org>.
- Hyperband paper: Li et al., 2017.
- scikit-learn model_selection: <https://scikit-learn.org/stable/modules/model_selection.html>.

---

> 📚 **Adaptación al español** de la lección "[Hyperparameter Tuning]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
