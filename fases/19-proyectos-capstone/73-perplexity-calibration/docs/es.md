# 73 — Perplexity y calibration

> Perplexity: exp(avg cross-entropy). Lower = mejor. Standard para LM eval. Calibration: ECE (Expected Calibration Error), reliability. Confidence vs accuracy.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/71
**Tiempo estimado:** ~20 minutos

## Objetivos

- Perplexity.
- ECE.
- Reliability diagram.
- Calibration methods.

## Constrúyelo

```python
import numpy as np


def perplexity(losses):
    """Perplexity = exp(mean loss)."""
    return np.exp(np.mean(losses))


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: weighted avg |accuracy - confidence|."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (labels[mask] == 1).mean()
        conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return ece
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-perplexity-calibration
fase: 19
leccion: 73
---

1. Perplexity.
2. ECE.
3. Reliability diagram.
4. Temperature scaling.
```

## Ejercicios

1. **PPL**: WikiText-103.
2. **ECE**: 1000 samples.
3. **Desafío**: temp
   scaling.

## Lecturas recomendadas

- "Calibration" (Guo 2017)
- "Temperature Scaling"
  (Guo 2017)
- "WikiText" (Merity 2016)



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
> "[73-perplexity-calibration]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
