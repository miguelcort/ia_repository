# 71 — Classical metrics

> Classical metrics: exact match, F1, BLEU, ROUGE, METEOR. Para text generation, QA, summarization, translation. Standard en benchmarks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/27, 19/70
**Tiempo estimado:** ~20 minutos

## Objetivos

- Exact match / F1.
- BLEU / ROUGE.
- METEOR.
- SacreBLEU.

## Constrúyelo

```python
def exact_match(pred, target):
    return int(pred.strip().lower() == target.strip().lower())


def f1_score(pred, target):
    pred_tokens = pred.lower().split()
    target_tokens = target.lower().split()
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(target_tokens)
    return 2 * prec * rec / (prec + rec)


def bleu_score(pred, target, max_n=4):
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([target.split()], pred.split(),
                        max_n)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-classical-metrics
fase: 19
leccion: 71
---

1. EM / F1.
2. BLEU / ROUGE.
3. METEOR.
4. SacreBLEU.
```

## Ejercicios

1. **F1**: 100 QA pairs.
2. **BLEU**: 100
   translations.
3. **Desafío**: custom
   metric.

## Lecturas recomendadas

- "Squad" (Rajpurkar 2016)
- "SacreBLEU" (Post 2018)
- "ROUGE" (Lin 2004)



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
> "[71-classical-metrics]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
