# 33 — Multihead self-attention

> Multihead self-attention: Q, K, V projections, scaled dot-product, multi-head, causal mask (GPT). Implementación: PyTorch nn.MultiheadAttention o desde cero. FlashAttention (Dao 2022) para memory-efficient.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/32
**Tiempo estimado:** ~25 minutos

## Objetivos

- Multihead attention.
- Causal mask.
- FlashAttention integration.
- Benchmark.

## Constrúyelo

```python
import torch
import torch.nn.functional as F


def multihead_attention(x, W_qkv, W_o, n_heads, causal=True):
    """Multihead self-attention."""
    B, T, D = x.shape
    qkv = x @ W_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    H = n_heads
    Dh = D // H
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    v = v.view(B, T, H, Dh).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v,
                                          is_causal=causal)
    return out.transpose(1, 2).reshape(B, T, D) @ W_o
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mha
fase: 19
leccion: 33
---

1. QKV projection.
2. Multi-head.
3. Causal mask.
4. FlashAttention.
5. SDPA.
```

## Ejercicios

1. **MHA**: 8 heads,
   d=512.
2. **Causal**: GPT-style.
3. **Desafío**: FlashAttn
   16K context.

## Detalles

MHA: Q,K,V proyectados, split en H cabezas (cada una
de dim D/H), atención paralela, concat, output
projection. Multi-head permite aprender diferentes
relaciones por cabeza (sintácticas, semánticas,
coreferencia).

GQA (Grouped-Query Attention, Ainslie 2023, Llama 2-3,
Mistral): K,V shared entre grupos de heads. Reduce
KV cache memory 4-8x en inference, mantiene quality.

MQA (Multi-Query, Shazeer 2019): single K, V head
shared. Más agresivo. Faster pero quality drop.

FlashAttention 2 (Dao 2023): memory-efficient via
tiling. Memory O(N) vs O(N²). Standard en producción.

PyTorch SDPA (torch.nn.functional.scaled_dot_product_
attention): unified interface que selecciona
implementation automáticamente (Flash, mem-efficient,
math). Default en HF Transformers.

Causal mask: triangular superior. Prevents attending
to future. Pre-compute una vez, add a scores.

Sliding window (Mistral): local attention window
W=4096. Reduce compute, mantiene quality para
long-context.

Hoy: MHA + RoPE + GQA + FlashAttn 2 + sliding window
= Llama 3, Mistral, Qwen 2.5.

## Lecturas recomendadas

- "Attention Is All You Need" (2017)
- "FlashAttention 2" (Dao 2023)
- "GQA" (Ainslie 2023)
- "PyTorch SDPA" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[33-multihead-self-attention]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
