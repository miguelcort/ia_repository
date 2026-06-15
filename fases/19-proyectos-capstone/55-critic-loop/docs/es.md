# 55 — Critic loop

> Critic loop: el LLM evalúa su propio output, identifica weaknesses, propone revisions. Iterar hasta quality threshold. Frameworks: Constitutional AI, Self-Refine, CRITIC.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/54
**Tiempo estimado:** ~20 minutos

## Objetivos

- Self-critique.
- Quality threshold.
- Iterative refinement.
- Convergence.

## Constrúyelo

```python
def critic_loop(generator, critic, prompt,
              max_iters=3, threshold=0.8):
    """Generate -> critique -> refine loop."""
    output = generator(prompt)
    for i in range(max_iters):
        score, critique = critic(output)
        if score >= threshold:
            return output
        output = generator(f"Refine: {prompt}\n"
                          f"Critique: {critique}")
    return output
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-critic-loop
fase: 19
leccion: 55
---

1. Generate output.
2. Critic score.
3. Refine.
4. Converge.
```

## Ejercicios

1. **Loop**: 5 iterations.
2. **Threshold**: 0.9.
3. **Desafío**: critic
   con few-shot.

## Lecturas recomendadas

- "Self-Refine" (Madaan 2023)
- "CRITIC" (Gou 2024)
- "Constitutional AI" (Bai 2022)



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
> "[55-critic-loop]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
