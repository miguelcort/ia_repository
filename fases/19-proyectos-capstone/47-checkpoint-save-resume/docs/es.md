# 47 — Checkpoint save/resume

> Checkpoint: save model, optimizer, scheduler, scaler, step, rng state. Resume desde último step. Atomic save (write to tmp, rename). Sharding para FSDP/ZeRO.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/36
**Tiempo estimado:** ~20 minutos

## Objetivos

- Save model + optimizer + state.
- Resume from checkpoint.
- Atomic save.
- Best-K tracking.

## Constrúyelo

```python
import torch


def save_checkpoint(model, optimizer, scheduler, step,
                  loss, path):
    state = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            if scheduler else None,
            "step": step, "loss": loss}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    import os
    os.replace(tmp, path)


def load_checkpoint(model, optimizer, scheduler, path):
    state = torch.load(path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state["step"]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-checkpoint
fase: 19
leccion: 47
---

1. Save model + opt + state.
2. Resume.
3. Atomic write.
4. Best-K.
```

## Ejercicios

1. **Save**: every 1000
   steps.
2. **Resume**: continue
   training.
3. **Desafío**: sharded
   checkpoint FSDP.

## Detalles

Atomic save: write a .tmp, os.replace (atomic rename).
Previene corrupción si crash mid-write. Hugging Face
usa este pattern.

Save state: (1) model.state_dict() (fp32 master copy).
(2) optimizer.state_dict() (AdamW: m, v, step, etc).
(3) scheduler.state_dict() (LR, step, etc). (4) step
count. (5) rng_state (PyTorch, NumPy, Python, CUDA).
(6) scaler.state_dict() (AMP). (7) best_metric.

FSDP checkpoint (PyTorch 2024): sharded state_dict,
cada rank guarda su shard. `torch.distributed.fsdp.
fully_sharded_summon_full_params` para gather.

Safetensors: alternative a pickle. Sin código malicioso,
faster load, cross-language. Standard en HF.

Streaming save: save to S3/GCS, evita disk space.
Async save thread.

Resume: load state, optimizer.load_state_dict(),
scheduler.load_state_dict(), rng state restore. Sin
rng restore: no-determinism en data shuffle.

Best-K: track val loss, save top-K checkpoints. Útil
para early stopping.

Hoy: save cada 1000 steps + best-K=3. FSDP con
sharded state dict. safetensors format.

## Lecturas recomendadas

- "PyTorch save/load" (2024)
- "FSDP Checkpoint" (PyTorch 2024)
- "Hugging Face save_strategy"
  (2024)
- "safetensors" (2022)

---

> 📚 **Adaptación al español** de la lección
> "[47-checkpoint-save-resume]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
