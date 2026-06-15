# 44 — Cosine LR schedule con warmup

> Cosine LR (Loshchilov 2017): linear warmup (1-10% steps) → cosine decay a min_lr. Standard en pre-training. Variantes: WSD (warmup-stable-decay), trapezoidal, inverse sqrt.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/36
**Tiempo estimado:** ~20 minutos

## Objetivos

- Linear warmup.
- Cosine decay.
- WSD schedule.
- Compare con constant.

## Constrúyelo

```python
import math


def lr_schedule(step, warmup, max_steps, max_lr, min_lr=0):
    """Cosine schedule con warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(
        math.pi * progress))


def wsd_schedule(step, warmup, stable, max_steps, max_lr,
                min_lr=0):
    """WSD: warmup, stable, decay."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step < warmup + stable:
        return max_lr
    progress = (step - warmup - stable) / (
        max_steps - warmup - stable)
    return min_lr + (max_lr - min_lr) * (1 - progress)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-lr-schedule
fase: 19
leccion: 44
---

1. Linear warmup.
2. Cosine decay.
3. WSD.
4. Compare en val loss.
```

## Ejercicios

1. **Cosine**: 1K steps.
2. **WSD**: 1K steps.
3. **Desafío**: find best
   warmup ratio.

## Detalles

Cosine schedule (Loshchilov 2017, SGDR): lr(step) =
0.5 * (max_lr - min_lr) * (1 + cos(π * step / T_max))
+ min_lr. Smooth decay, no sudden drops. Standard en
pre-training. min_lr = 0.1 * max_lr típico.

Warmup: linear ramp 0 → max_lr en primeros 1-10% steps.
Previene early instability. 1% para large batch, 10%
para small batch. 2000-5000 steps en pre-training.

WSD (warmup-stable-decay, Hu 2024): (1) warmup 1%.
(2) Stable phase (constant max_lr) por 80%. (3) Decay
20% usando sqrt o cosine. Más flexible que cosine
puro, no requiere knowing max_steps upfront.

Inverse sqrt (Vaswani 2017): lr = d_model^(-0.5) *
min(step^(-0.5), step * warmup^(-1.5)). Usado en
original transformer. Menos usado hoy.

Trapezoidal: similar a WSD. Implementado en nanoGPT.

Constant: simple, no schedule. Útil para SFT, no para
pre-training (waste compute al final).

Hoy: cosine con warmup 1-5% es el standard en
pre-training (Llama 3, Mistral). WSD ganando tracción
por flexibility.

Mini-cycles: SGDR mini-cycles, restart cada N steps.
Para fine-tuning ayuda a escapar local minima.

## Lecturas recomendadas

- "SGDR" (Loshchilov 2017)
- "WSD" (Hu 2024)
- "Scaling Laws" (Kaplan 2020)
- "Llama 3" (Meta 2024)

---

> 📚 **Adaptación al español** de la lección
> "[44-cosine-lr-warmup]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
