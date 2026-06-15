# 76 — Collective ops from scratch

> Collective ops (all-reduce, all-gather, reduce-scatter, broadcast) para distributed training. NCCL backend, ring/tree algorithms. Implementación: torch.distributed, custom.

**Tipo:** Construir
**Lenguajes:** Python, CUDA (opcional)
**Prerrequisitos:** Fase 19/48
**Tiempo estimado:** ~30 minutos

## Objetivos

- All-reduce, all-gather.
- Ring algorithm.
- NCCL.
- Benchmark bandwidth.

## Constrúyelo

```python
import torch
import torch.distributed as dist


def all_reduce(tensor):
    """All-reduce: sum across all ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def all_gather(tensor):
    """All-gather: collect from all ranks."""
    world = dist.get_world_size()
    out = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor)
    return out


def ring_all_reduce(tensor, rank, world_size):
    """Ring all-reduce (simulated)."""
    size = tensor.numel()
    chunks = list(torch.chunk(tensor, world_size, dim=0))
    # Reduce-scatter + all-gather
    for step in range(world_size - 1):
        send = (rank - step) % world_size
        recv = (rank - step - 1) % world_size
        chunks[recv] = chunks[recv] + chunks[send]
    return torch.cat(chunks)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-collectives
fase: 19
leccion: 76
---

1. All-reduce.
2. All-gather.
3. Reduce-scatter.
4. Ring algorithm.
```

## Ejercicios

1. **All-reduce**: 8 GPUs.
2. **Ring**: sim 4 nodes.
3. **Desafío**: bandwidth
   benchmark.

## Lecturas recomendadas

- "NCCL" (NVIDIA 2016)
- "Ring AllReduce"
  (Andrews 2017)
- "PyTorch distributed"
  (2024)



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
> "[76-collective-ops-from-scratch]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
