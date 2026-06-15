# 36 — Training loop y eval

> Training loop: forward → cross-entropy loss → backward → clip grad → AdamW step → log. Eval: validation pass per N steps, perplexity. nanoGPT: ~300 líneas, entrenar GPT-2 small en 1 GPU.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/35
**Tiempo estimado:** ~30 minutos

## Objetivos

- Training loop.
- Cross-entropy + perplexity.
- Eval pass.
- nanoGPT scale.

## Constrúyelo

```python
import torch.nn.functional as F


def train_step(model, batch, optimizer, grad_clip=1.0):
    """Single training step."""
    ids, targets = batch
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                          targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def eval_pass(model, val_loader):
    """Validation pass: perplexity."""
    total_loss = 0
    n = 0
    for ids, targets in val_loader:
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
        total_loss += loss.item() * ids.size(0)
        n += ids.size(0)
    return {"perplexity": (total_loss / n) ** 0.5}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-train-loop
fase: 19
leccion: 36
---

1. Cross-entropy.
2. Backward.
3. Grad clip.
4. AdamW.
5. Perplexity.
```

## Ejercicios

1. **Loop**: 1000 steps
   en TinyShakespeare.
2. **Eval**: medir ppl.
3. **Desafío**: GPT-2
   small en 1 GPU.

## Lecturas recomendadas

- "nanoGPT" (Karpathy 2022)
- "AdamW" (Loshchilov 2019)
- "Cross-entropy" (Info theory)

---

> 📚 **Adaptación al español** de la lección
> "[36-training-loop-eval]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
