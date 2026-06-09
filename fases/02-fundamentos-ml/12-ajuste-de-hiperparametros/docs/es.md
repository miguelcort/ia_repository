# Ajuste de hiperparametros

> La diferencia entre un modelo mediocre y uno de produccion esta a menudo en 20 minutos de tuning bien hecho.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-evaluacion-de-modelos-y-overfitting
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar k-fold cross-validation.
- Implementar grid search.
- Implementar random search.
- Diagnosticar cuando usar cada estrategia.

## Constrúyelo

```python
def k_folds(n, k=5, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tuning
fase: 02
leccion: 12
---

1. Pocos hiperparametros: Grid.
2. Muchos: Random/Optuna.
3. Score final: nested CV.
```

## Ejercicios

1. **Bayesian**: implementa un surrogate model (random forest) y
   el criterio de mejora esperada.
2. **Hyperband**: implementa el bracket de sucesion con pruner.
3. **Desafio**: implementa Optuna-TPE simplificado.

## Lecturas recomendadas

- "Random Search for Hyper-Parameter Optimization" (Bergstra & Bengio, 2012)
- "Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011)
- Optuna: <https://optuna.org/>

---

> 📚 **Adaptación al español** de la lección "[Hyperparameter Tuning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).