# 53 — Result evaluator

> Result evaluator: análisis estadístico de resultados experimentales. Significance testing, effect size, confidence intervals, multiple comparisons. AutoML eval con statistical rigor.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/52
**Tiempo estimado:** ~20 minutos

## Objetivos

- Statistical tests.
- Effect size.
- Bootstrap CI.
- Multiple comparison correction.

## Constrúyelo

```python
import numpy as np
from scipy import stats


def compare_models(metric_a, metric_b, alpha=0.05):
    """Statistical comparison."""
    t_stat, p_value = stats.ttest_rel(metric_a, metric_b)
    # Effect size (Cohen's d)
    d = (np.mean(metric_a) - np.mean(metric_b)) / np.std(
        metric_b)
    # Bootstrap CI
    diffs = np.array(metric_a) - np.array(metric_b)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return {"p_value": p_value, "effect_size": d,
            "ci_95": (ci_low, ci_high),
            "significant": p_value < alpha}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-eval-results
fase: 19
leccion: 53
---

1. T-test / Wilcoxon.
2. Effect size (Cohen's d).
3. Bootstrap CI.
4. Bonferroni / BH.
```

## Ejercicios

1. **Compare**: 2 modelos
   en 10 seeds.
2. **Effect size**: 30+
   runs.
3. **Desafío**: Bonferroni
   correction.

## Lecturas recomendadas

- "Statistical Methods in
  ML" (Dietterich 1998)
- "Bootstrap" (Efron 1979)
- "BH correction" (Benjamini 1995)



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
> "[53-result-evaluator]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
