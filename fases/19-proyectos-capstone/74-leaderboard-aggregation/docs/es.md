# 74 — Leaderboard aggregation

> Leaderboard aggregation: combinar scores de múltiples benchmarks. Promedio simple, ponderado, ranking (Bradley-Terry), Elo. Frameworks: Hugging Face OpenLLM, Artificial Analysis, LMSYS.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/71-73
**Tiempo estimado:** ~20 minutos

## Objetivos

- Average aggregation.
- Weighted score.
- Bradley-Terry ranking.
- Elo.

## Constrúyelo

```python
def average_score(model_results):
    """Simple avg across benchmarks."""
    return sum(model_results.values()) / len(model_results)


def weighted_score(model_results, weights):
    """Weighted sum of scores."""
    return sum(model_results[k] * w for k, w in weights.items())


def elo_update(rating_a, rating_b, score_a, k=32):
    """Elo rating (LMSYS Arena)."""
    exp_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-leaderboard
fase: 19
leccion: 74
---

1. Simple avg.
2. Weighted.
3. Bradley-Terry.
4. Elo.
```

## Ejercicios

1. **Avg**: 5 benchmarks.
2. **BT**: 10 models.
3. **Desafío**: custom
   leaderboard.

## Lecturas recomendadas

- "Open LLM Leaderboard"
  (Hugging Face 2024)
- "LMSYS Arena" (Zheng 2023)
- "Bradley-Terry" (1952)



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
> "[74-leaderboard-aggregation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
