# 50 — Hypothesis generator

> Hypothesis generator: el agente propone hipótesis research. LLM + chain-of-thought + prior literature. Criterios: novelty, feasibility, impact. Loop: generate → critique → refine.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/05, Fase 14
**Tiempo estimado:** ~25 minutos

## Objetivos

- LLM hypothesis generation.
- Novelty/feasibility scoring.
- Critique loop.
- Research output.

## Constrúyelo

```python
def generate_hypothesis(topic, prior_lit, llm):
    prompt = f"""Topic: {topic}
Prior literature: {prior_lit}

Generate 3 testable hypotheses with:
- Novelty (not in prior work)
- Feasibility (data + compute available)
- Impact (potential contribution)

For each: claim, test, expected outcome."""
    return llm.generate(prompt)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-hypothesis
fase: 19
leccion: 50
---

1. Topic + prior lit.
2. Hypothesis generation.
3. Novelty check.
4. Feasibility.
5. Critique loop.
```

## Ejercicios

1. **Generate**: 5
   hipótesis para un
   dataset.
2. **Critique**: implementar
   novelty check.
3. **Desafío**: end-to-end
   research loop.

## Lecturas recomendadas

- "AI Co-scientist" (Google 2024)
- "STORM" (Shao 2024)
- "SciMON" (Wang 2024)



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
> "[50-hypothesis-generator]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
