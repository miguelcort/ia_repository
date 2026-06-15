# 86 — Constitutional rules engine

> Constitutional rules engine: codified rules (constitution) que el LLM debe seguir. Anthropic CAI. Rule-based + LLM judge hybrid. Eval compliance rate.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 18/05, 19/15
**Tiempo estimado:** ~20 minutos

## Objetivos

- Constitution definition.
- Rule engine.
- LLM judge.
- Compliance eval.

## Constrúyelo

```python
CONSTITUTION = [
    "Do not provide harmful information",
    "Respect user privacy",
    "Be honest about uncertainty",
    "Refuse illegal requests",
    "Avoid biased responses",
]


def check_compliance(response, constitution,
                  judge_llm):
    """Check response vs constitution."""
    for rule in constitution:
        score = judge_llm(f"Rule: {rule}\nResponse: {response}\n"
                         f"Does it comply?")
        if not score:
            return False, rule
    return True, None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rules-engine
fase: 19
leccion: 86
---

1. Constitution.
2. Rule engine.
3. LLM judge.
4. Compliance eval.
```

## Ejercicios

1. **10 rules**.
2. **Eval**: 100 cases.
3. **Desafío**: 95%
   compliance.

## Lecturas recomendadas

- "Constitutional AI"
  (Bai 2022)
- "Anthropic Constitution"
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
> "[86-constitutional-rules-engine]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
