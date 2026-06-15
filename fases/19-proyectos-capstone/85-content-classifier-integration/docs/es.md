# 85 — Content classifier integration

> Content classifier integration: integrar LlamaGuard, OpenAI Moderation, custom classifiers en LLM pipeline. Pre-LLM (input filter) y post-LLM (output filter). Latency overhead, F1, throughput.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/15, 19/29
**Tiempo estimado:** ~20 minutos

## Objetivos

- LlamaGuard integration.
- OpenAI Moderation.
- Pre/post LLM filter.
- Latency tracking.

## Constrúyelo

```python
def moderation_pipeline(prompt, llm, llamaguard,
                      openai_mod):
    """Pipeline: input + output moderation."""
    # Pre-LLM
    if llamaguard.is_unsafe(prompt) or openai_mod.flagged(prompt):
        return {"blocked": True, "stage": "input"}
    # LLM
    response = llm(prompt)
    # Post-LLM
    if llamaguard.is_unsafe(response) or openai_mod.flagged(response):
        return {"blocked": True, "stage": "output"}
    return {"blocked": False, "response": response}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-content-classifier
fase: 19
leccion: 85
---

1. LlamaGuard.
2. OpenAI Moderation.
3. Pre/post filter.
4. Latency.
```

## Ejercicios

1. **Pipeline**: 1K
   requests.
2. **Latency**: <50ms.
3. **Desafío**: ASR <
   1%, latency <100ms.

## Lecturas recomendadas

- "LlamaGuard 3" (Meta 2024)
- "OpenAI Moderation"
  (2024)
- "guardrails-ai" (2024)



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
> "[85-content-classifier-integration]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
