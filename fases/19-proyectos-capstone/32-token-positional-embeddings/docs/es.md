# 32 — Token + positional embeddings

> Embeddings: token (vocab × d_model), positional (sinusoidal, RoPE, ALiBi). RoPE (Su 2021, GPT-NeoX, Llama): rotate Q,K per position, extrapolation. ALiBi (Press 2022, BLOOM): linear bias.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/30
**Tiempo estimado:** ~25 minutos

## Objetivos

- Token embedding.
- RoPE (rotary position).
- Sinusoidal / learned.
- ALiBi bias.

## Constrúyelo

```python
import numpy as np


def sinusoidal_pos_emb(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angles = pos / 10000 ** (i / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def apply_rope(x, freqs_cis):
    """Rotary position embedding."""
    # x: (B, H, T, D)
    cos = np.cos(freqs_cis)
    sin = np.sin(freqs_cis)
    x_rot = np.stack([-x[..., 1::2], x[..., ::2]], axis=-1)
    x_rot = x_rot.reshape(x.shape)
    return x * cos + x_rot * sin
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-pos-emb
fase: 19
leccion: 32
---

1. Token embedding.
2. RoPE (rotary).
3. Sinusoidal.
4. ALiBi.
5. Long-context.
```

## Ejercicios

1. **Sinusoidal**: position
   0 a 2048.
2. **RoPE**: apply to Q, K.
3. **Desafío**: ALiBi
   with 32K context.

## Tipos de positional encoding

Sinusoidal (Vaswani 2017): PE(pos, 2i) = sin(pos /
10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).
No learned, extrapolates beyond train length. Used in
original transformer.

Learned absolute: embedding per position. Simple, no
extrapolation. GPT-2, GPT-3.

RoPE (Su 2021, GPT-NeoX, Llama 1-3, Mistral): rotate
Q, K per position. Q'_i = Q_i · cos(i·θ) + rotated(Q_i)
· sin(i·θ). Cada dimensión tiene su frequency θ_i =
10000^(-2i/d). Extrapolation via NTK-aware scaling or
YaRN.

ALiBi (Press 2022, BLOOM): linear bias to attention
scores: score[i,j] = QK^T - m·|i-j|, m slope per head.
Simple, efficient, extrapolates.

YaRN (Peng 2023): RoPE + NTK scaling + attention
temperature. Llama 2 7B → 128K context with YaRN.

CoPE (Meta 2024): position per token, no per-dimension.
Flexible: continuous positions, sentence-level.

Hoy: RoPE es el standard (Llama 3, Mistral, Qwen). Con
YaRN/RoPE-abacus para long-context (128K-1M).

## Lecturas recomendadas

- "RoPE" (Su 2021)
- "ALiBi" (Press 2022)
- "Sinusoidal" (Vaswani 2017)
- "YaRN" (Peng 2023)
- "CoPE" (Meta 2024)

---

> 📚 **Adaptación al español** de la lección
> "[32-token-positional-embeddings]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
