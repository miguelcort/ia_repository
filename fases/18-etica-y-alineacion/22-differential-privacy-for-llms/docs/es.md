# 22 — Differential privacy para LLMs

> Differential privacy (DP) para LLMs: DP-SGD (Abadi 2016), DP fine-tuning, PATE. Garantiza que la presencia de un training example no afecta significativamente el output. ε越小，越隐私。

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/20, 18/21
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir differential privacy (ε, δ).
- Implementar DP-SGD.
- Aplicar a LLM fine-tuning.
- Diagnosticar privacy-utility trade-off.

## Constrúyelo

```python
import numpy as np


def clip_gradients(grads, clip_norm=1.0):
    """Per-example gradient clipping para DP-SGD."""
    norm = np.linalg.norm(grads)
    if norm > clip_norm:
        grads = grads * clip_norm / norm
    return grads


def add_noise(grads, sigma, clip_norm=1.0):
    """Gaussian noise: σ * clip * I. (ε, δ)-DP guarantee."""
    noise = np.random.normal(0, sigma * clip_norm, grads.shape)
    return grads + noise


def dp_sgd_step(grads, lr=0.01, clip_norm=1.0, sigma=1.0):
    """DP-SGD: clip + noise + descent."""
    clipped = np.array([clip_gradients(g, clip_norm) for g in grads])
    noisy = add_noise(clipped.mean(axis=0), sigma, clip_norm)
    return -lr * noisy
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dp-llm
fase: 18
leccion: 22
---

1. DP-SGD con per-example clipping.
2. Gaussian noise para (ε, δ)-DP.
3. Privacy budget tracking.
4. Trade-off con utility.
```

## Ejercicios

1. **DP-SGD**: implementar en classifier.
2. **Privacy budget**: calcular ε acumulado.
3. **Desafío**: DP fine-tuning de Llama
   con Opacus.

## Lecturas recomendadas

- "Deep Learning with Differential Privacy"
  (Abadi 2016)
- "Opacus: PyTorch DP library" (Meta)
- "PATE" (Papernot 2017)

---

> 📚 **Adaptación al español** de la lección
> "[22-differential-privacy-for-llms]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
