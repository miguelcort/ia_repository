# 81 — End-to-end distributed train

> End-to-end distributed training: DDP + FSDP + pipeline + AMP + grad accum + checkpoint + monitoring. Llama 3 70B en 256 GPUs: ~2 weeks. Frameworks: torchrun, accelerate, deepspeed, Megatron.

**Tipo:** Capstone
**Lenguajes:** Python, Bash
**Prerrequisitos:** Fase 19/48, 19/76-80
**Tiempo estimado:** 30 horas

## Objetivos

- Full distributed setup.
- Llama 3 70B training.
- Multi-node.
- Monitoring.

## Constrúyelo

```python
"""Launch: torchrun --nproc_per_node=8 train.py"""
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    # Setup FSDP model
    model = setup_fsdp_zero3(Llama3_70B())
    # Data
    loader = setup_dataloader(rank, world)
    # Train
    for epoch in range(epochs):
        for batch in loader:
            loss = train_step(model, batch)
            if rank == 0 and step % 100 == 0:
                log_to_wandb({"loss": loss})
            if step % 1000 == 0:
                save_sharded(model, f"ckpt-{step}/")
```

## Úsalo

```bash
cd code
torchrun --nproc_per_node=8 main.py
```

## Despliégalo

```markdown
---
name: prompt-distributed-train
fase: 19
leccion: 81
---

1. DDP + FSDP.
2. AMP bf16.
3. Grad accum.
4. Sharded ckpt.
5. W&B.
```

## Ejercicios

1. **8 GPU**: DDP.
2. **8 GPU**: FSDP.
3. **Desafío**: 64 GPU
   Llama 3 8B.

## Lecturas recomendadas

- "PyTorch FSDP" (2024)
- "DeepSpeed" (Microsoft)
- "Megatron-LM" (Shoeybi 2019)
- "Llama 3" (Meta 2024)



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
> "[81-end-to-end-distributed-train]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
