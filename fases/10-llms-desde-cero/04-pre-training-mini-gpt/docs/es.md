# Pre-training mini-GPT

> Pre-training LLM: corpus masivo (10B-15T tokens, FineWeb, RedPajama-v2, Dolma), next-token prediction causal, AdamW (betas 0.9, 0.95, wd 0.1), cosine LR decay con warmup 1-10%, peak 6e-4, bf16 mixed precision, Flash Attention, gradient clipping max norm 1.0. Loop: forward, cross-entropy loss, backward, clip, AdamW step, log. Llama 3 405B: 15T tokens, 16K H100, 1-2 meses. nanoGPT: minimal GPT-2 en 300 líneas (Karpathy). Frameworks: PyTorch, Megatron-LM, DeepSpeed, FairScale.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/14-construye-un-transformer-capstone, 10/03-pipelines-de-datos
**Tiempo estimado:** ~35 minutos

## Objetivos

- Implementar cross-entropy y perplexity.
- Implementar cosine LR schedule con warmup.
- Implementar AdamW update.
- Implementar grad clip.
- Diagnosticar hiperparámetros SOTA.

## Constrúyelo

```python
def lr_schedule(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * decay_ratio))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pretraining
fase: 10
leccion: 04
---

1. Next-token prediction.
2. AdamW, cosine LR.
3. bf16, Flash Attention.
4. nanoGPT reference.
5. Llama 3 405B scale.
```

## Ejercicios

1. **nanoGPT**: implementar mini-GPT
   end-to-end.
2. **Distributed**: configurar
   DDP en multi-GPU.
3. **Desafio**: pre-entrenar
   GPT-2 small en 1 GPU.

## Lecturas recomendadas

- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3
- "nanoGPT" (Karpathy, 2022)
- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
- "DeepSpeed" (Microsoft)

---

> 📚 **Adaptación al español** de la lección "[Pre Training Mini GPT]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).