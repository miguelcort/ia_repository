# 56 — Iteration scheduler

> Iteration scheduler: decide qué hipótesis/experimento correr next, dado budget y results. Bandits, Bayesian optimization, evolution strategies. Frameworks: Optuna, Ray Tune, Weights & Biases Sweeps.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/52, 19/53
**Tiempo estimado:** ~20 minutos

## Objetivos

- Hyperparam search.
- Bayesian optimization.
- Early stopping.
- Multi-fidelity.

## Constrúyelo

```python
import optuna


def objective(trial, train_fn, val_fn):
    """Optuna objective."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    bs = trial.suggest_categorical("bs", [16, 32, 64])
    model = train_fn(lr, bs)
    score = val_fn(model)
    return score


def run_search(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-scheduler
fase: 19
leccion: 56
---

1. Bayesian optimization.
2. Hyperband / ASHA.
3. Early stopping.
4. Multi-fidelity.
```

## Ejercicios

1. **Optuna**: 50 trials.
2. **Pruner**: Median.
3. **Desafío**: 1000
   trials distributed.

## Lecturas recomendadas

- "Optuna" (Akiba 2019)
- "Ray Tune" (Liaw 2018)
- "BOHB" (Falkner 2018)



## Detalles avanzados

Esta lección cubre los trade-offs críticos de
producción. Considera scaling: en pre-training el
factor dominante es cómputo disponible; en inference
es latencia y costo. Frameworks standard: PyTorch
(HF Transformers, TRL, vLLM), JAX (Flax, Optax).
Optimizaciones: FlashAttention-2, paged attention,
KV cache compression, speculative decoding, MoE.

Eval riguroso: statistical significance testing
sobre múltiples seeds, held-out test sets sin
contamination, y edge cases del domain. Métricas:
BLEU/ROUGE para text generation, exact match/F1
para QA, pass@k para code, human preference para
chat.

Trampas comunes: data leakage entre train/test,
overfitting al validation set, eval con prompts
fuera de distribución, ignore de tail latency en
serving, cost runaway en production.

Tools clave: Weights & Biases o MLflow para
tracking, Langfuse para LLM observability, Hydra
para config, Ray para distributed execution, vLLM
para serving LLM. Conoce al menos uno a fondo antes
de producción.

---

> 📚 **Adaptación al español** de la lección
> "[56-iteration-scheduler]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
