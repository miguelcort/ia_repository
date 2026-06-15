# 48 — Distributed FSDP / DDP

> Distributed training: DDP (data parallel, full replica per GPU), FSDP (sharded params + grads + optim, ZeRO-3). Mixed precision, gradient compression, overlap. Frameworks: torch.distributed, DeepSpeed, FairScale, Accelerate.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/45-47
**Tiempo estimado:** ~30 minutos

## Objetivos

- DDP setup.
- FSDP sharding.
- Mixed precision.
- Benchmark.

## Constrúyelo

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
                as FSDP
from torch.distributed.fsdp import MixedPrecision


def fsdp_setup(model, use_mp=True):
    """FSDP wrapping con mixed precision."""
    mp_policy = (MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16) if use_mp else None)
    return FSDP(model, mixed_precision=mp_policy,
               use_orig_params=True)


def ddp_setup(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-distributed
fase: 19
leccion: 48
---

1. DDP setup.
2. FSDP sharding.
3. Mixed precision.
4. Benchmark scaling.
```

## Ejercicios

1. **DDP**: 4 GPUs.
2. **FSDP**: 8 GPUs,
   Llama 3 70B.
3. **Desafío**: 256 GPU
   training.

## Lecturas recomendadas

- "PyTorch FSDP" (2024)
- "ZeRO" (Rajbhandari 2020)
- "DeepSpeed" (Microsoft 2020)
- "Accelerate" (Hugging Face)

## Detalles adicionales

Mixed DDP + FSDP: DDP outer (8 nodes), FSDP inner
(8 GPUs/node). 64 GPU effective, model sharded dentro
de node, replicated across nodes.

Init: `torch.distributed.init_process_group("nccl")`.
Backend: NCCL (GPU), Gloo (CPU). Launcher:
torchrun, accelerate, deepspeed, slurm.

Hoy: FSDP + bf16 + grad accumulation es el standard
para training > 13B params. DeepSpeed para > 100B
(con offload).

---

> 📚 **Adaptación al español** de la lección
> "[48-distributed-fsdp-ddp]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
