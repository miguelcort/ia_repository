# 77 — Data parallel (DDP)

> DDP (Data Parallel): cada GPU replica el modelo, data partitioned across ranks. Forward, backward, all-reduce grads, step. PyTorch nn.parallel.DistributedDataParallel. Synchronized BN, gradient bucketing.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/48, 19/76
**Tiempo estimado:** ~25 minutos

## Objetivos

- DDP setup.
- DDP wrapping.
- Distributed sampler.
- Synchronized BN.

## Constrúyelo

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_train(rank, world_size, model_fn, dataset):
    dist.init_process_group("nccl", rank=rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)
    model = model_fn().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(dataset,
                                          sampler=sampler)
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in loader:
            loss = ddp_model(batch)
            loss.backward()
            ddp_model.step()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ddp
fase: 19
leccion: 77
---

1. DDP setup.
2. Distributed sampler.
3. All-reduce grads.
4. Sync BN.
```

## Ejercicios

1. **DDP**: 4 GPUs.
2. **Sampler**: shuffle
   partition.
3. **Desafío**: 32 GPU
   scaling.

## Lecturas recomendadas

- "PyTorch DDP" (2024)
- "DDP paper" (Li 2020)
- "torchrun" (2024)



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
> "[77-data-parallel-ddp]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
