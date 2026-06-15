# 45 — Gradient clipping y AMP

> Gradient clipping: max_norm=1.0 previene exploding gradients. AMP (Automatic Mixed Precision): fp16/bf16 forward/backward, fp32 weights/mom. bf16 preferred en GPUs modernos (no overflow).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/36
**Tiempo estimado:** ~20 minutos

## Objetivos

- Clip grad by norm.
- AMP / bf16.
- Compare memory, speed.
- Loss scaling (fp16).

## Constrúyelo

```python
import torch
from torch.cuda.amp import autocast, GradScaler


def train_step_amp(model, batch, optimizer, scaler=None):
    """Training step con AMP bf16."""
    ids, targets = batch
    with autocast(dtype=torch.bfloat16):
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-grad-amp
fase: 19
leccion: 45
---

1. Clip grad norm.
2. AMP bf16.
3. Compare fp32 vs bf16.
4. Memory savings.
```

## Ejercicios

1. **Clip**: max_norm 1.0
   vs 5.0.
2. **AMP**: medir speedup.
3. **Desafío**: fp16 con
   loss scaling.

## Detalles

Gradient clipping: previene exploding gradients en
training. max_norm=1.0 (PyTorch default, common in
LLM training). `torch.nn.utils.clip_grad_norm_` calcula
global norm, scales si > max_norm. Clipping by value
menos común.

bf16 (bfloat16, Kalamkar 2019): brain float, 7-bit
mantissa (vs fp16 10-bit). Rango igual a fp32
(mismo exponent). Sin loss scaling. Default en
Hopper/Ampere. PyTorch: `torch.bfloat16`.

fp16 (Micikevicius 2018, AMP): 10-bit mantissa, 5-bit
exponent. Overflow risk. Necesita loss scaling (Grad
Scaler) para mantener scale. Más precisión que bf16
pero más complejo.

Mixed precision: (1) Forward/backward en bf16/fp16.
(2) Weights, gradients, optimizer state en fp32 (master
copy). (3) Optimizer step en fp32. PyTorch AMP /
torch.amp.autocast manages this.

Memory savings: 2x (fp16/bf16 weights), 4x (8-bit
optim AdamW), 8x (4-bit + 8-bit AdamW).

Speedup: 2-3x en modern GPUs (Tensor Cores). Throughput
mejora dramáticamente en H100.

Hoy: bf16 + clip 1.0 + AdamW betas 0.9/0.95 + wd 0.1
es el standard. fp16 solo en legacy GPUs (V100, T4).

## Lecturas recomendadas

- "AMP" (Micikevicius 2018)
- "bf16" (Kalamkar 2019)
- "Grad Clip" (Pascanu 2013)
- "8-bit Adam" (Dettmers 2021)

---

> 📚 **Adaptación al español** de la lección
> "[45-gradient-clipping-amp]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
