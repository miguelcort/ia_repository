# 80 — Checkpoint sharded resume

> Sharded checkpoint: cada rank guarda su shard (FSDP/ZeRO-3). Resume re-construye modelo desde shards. Metadata: topology, layer specs. Frameworks: PyTorch FSDP, DeepSpeed.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/47, 19/78
**Tiempo estimado:** ~20 minutos

## Objetivos

- FSDP checkpoint.
- Sharded save/load.
- Resume from sharded.
- Resharding.

## Constrúyelo

```python
from torch.distributed.fsdp import FullyShardedDataParallel
                as FSDP
from torch.distributed.checkpoint import save, load


def save_sharded(model, path):
    """Save FSDP sharded checkpoint."""
    state = {"model": model.state_dict()}
    save(state, path)


def load_sharded(model, path):
    """Load FSDP sharded checkpoint."""
    state = {"model": model.state_dict()}
    load(state, path)
    model.load_state_dict(state["model"])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sharded-checkpoint
fase: 19
leccion: 80
---

1. FSDP sharded save.
2. Resume.
3. Resharding.
4. Format.
```

## Ejercicios

1. **Save**: 8 GPU FSDP.
2. **Resume**: 4 GPU.
3. **Desafío**: 256 GPU
   checkpoint.

## Lecturas recomendadas

- "PyTorch FSDP
  Checkpoint" (2024)
- "DeepSpeed Checkpoint"
  (Microsoft)
- "torch.distributed.checkpoint"
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
> "[80-checkpoint-sharded-resume]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
