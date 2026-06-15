# 79 — Pipeline parallel

> Pipeline parallelism (GPipe, PipeDream): split model en stages, cada stage en GPU diferente. Microbatches para overlap. Bubble overhead. Frameworks: torch.distributed.pipelining, DeepSpeed, Megatron.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/48, 19/78
**Tiempo estimado:** ~25 minutos

## Objetivos

- Pipeline stages.
- Microbatches.
- GPipe schedule.
- Bubble overhead.

## Constrúyelo

```python
def pipeline_forward(stages, micro_batch, n_microbatches=4):
    """GPipe: split batch en microbatches."""
    outputs = []
    for chunk in chunks(micro_batch, n_microbatches):
        x = chunk
        for stage in stages:
            x = stage(x)
        outputs.append(x)
    return outputs
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pipeline-parallel
fase: 19
leccion: 79
---

1. Stages.
2. Microbatches.
3. GPipe.
4. Bubble.
```

## Ejercicios

1. **GPipe**: 4 stages.
2. **Microbatch**: 8.
3. **Desafío**: 70B
   8x A100.

## Lecturas recomendadas

- "GPipe" (Huang 2019)
- "PipeDream" (Narayanan 2019)
- "Megatron" (Shoeybi 2019)



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
> "[79-pipeline-parallel]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
