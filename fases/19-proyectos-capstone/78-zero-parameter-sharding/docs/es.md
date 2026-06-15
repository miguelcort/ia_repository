# 78 — ZeRO parameter sharding

> ZeRO (Zero Redundancy Optimizer, Rajbhandari 2020): shard params/grads/optimizer state. ZeRO-1 (optimizer), ZeRO-2 (+grads), ZeRO-3 (+params, FSDP equivalent). Memory savings O(N/n_gpus).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/77, 19/48
**Tiempo estimado:** ~25 minutos

## Objetivos

- ZeRO-1/2/3.
- FSDP integration.
- Memory savings.
- Compare con DDP.

## Constrúyelo

```python
from torch.distributed.fsdp import FullyShardedDataParallel
                as FSDP
from torch.distributed.fsdp import ShardingStrategy


def setup_fsdp_zero3(model):
    """ZeRO-3 / FSDP sharding."""
    return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
               use_orig_params=True)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-zero-shard
fase: 19
leccion: 78
---

1. ZeRO-1 (opt).
2. ZeRO-2 (+grads).
3. ZeRO-3 (+params).
4. FSDP equivalent.
```

## Ejercicios

1. **FSDP**: 8 GPUs.
2. **Memory**: 70B model.
3. **Desafío**: 256 GPU
   Llama 3 405B.

## Lecturas recomendadas

- "ZeRO" (Rajbhandari 2020)
- "ZeRO-Infinity" (2021)
- "FSDP" (PyTorch 2024)



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
> "[78-zero-parameter-sharding]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
