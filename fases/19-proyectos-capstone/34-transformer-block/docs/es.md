# 34 — Transformer block

> Transformer block (pre-norm): MHA → residual → LayerNorm → MLP → residual → LayerNorm. Componentes: GeLU/SwiGLU activation, RoPE, GQA (grouped-query attention), KV cache.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/33
**Tiempo estimado:** ~25 minutos

## Objetivos

- Pre-norm block.
- SwiGLU MLP.
- GQA.
- Implementar block.

## Constrúyelo

```python
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x),
                         self.ln1(x), need_weights=False)[0]
        return x + self.mlp(self.ln2(x))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-block
fase: 19
leccion: 34
---

1. Pre-norm.
2. MHA + MLP.
3. Residual.
4. SwiGLU.
5. GQA.
```

## Ejercicios

1. **Block**: 1 layer.
2. **GQA**: 4 query, 1 KV.
3. **Desafío**: 12 layers
   Llama 3 arch.

## Detalles del block

Pre-norm: x' = x + Sublayer(LayerNorm(x)). Más
estable que post-norm. Default en GPT-2/3/4, LLaMA,
Mistral, Qwen. Pre-norm tiene residual stream que
crece; final LayerNorm al output.

Post-norm: x' = LayerNorm(x + Sublayer(x)). Original
del paper 2017. Requiere warmup cuidadoso.

SwiGLU MLP (Shazeer 2020, Llama 1-3): gate · up →
down, donde gate = Swish(W_gate·x), up = W_up·x. 3
linear projections vs 2 en standard. Better quality.

d_ff típicamente 2.67-4× d_model. Llama 2 7B:
d_model=4096, d_ff=11008 (2.6875×).

GQA (Ainslie 2023, Llama 2-3): K, V shared entre
grupos de Q heads. Llama 2 70B: 8 KV heads para 64
Q heads. Reduce KV cache 8x en inference.

Normalization variants: RMSNorm (Llama) vs LayerNorm.
RMSNorm más rápido (no mean centering). Default en
Llama, Mistral, Qwen.

Hoy: pre-norm + SwiGLU + RMSNorm + GQA + RoPE =
Llama 3, Mistral, Qwen 2.5.

## Lecturas recomendadas

- "Llama 2" (Touvron 2023)
- "GQA" (Ainslie 2023)
- "SwiGLU" (Shazeer 2020)
- "RMSNorm" (Zhang 2019)

---

> 📚 **Adaptación al español** de la lección
> "[34-transformer-block]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
