# Transformer completo

> Encoder block: self-attn + Add&Norm + FFN + Add&Norm. Decoder: masked self-attn + cross-attn + FFN. Pre-LN (moderno) más estable que Post-LN. Residual connections son el "gradient highway" que permite 100+ capas. Activations: GELU (BERT), SwiGLU (Llama/Mistral).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/04-positional-encoding
**Tiempo estimado:** ~35 minutos

## Objetivos

- Implementar LayerNorm, GELU, FFN.
- Implementar encoder block.
- Implementar decoder block con cross-attention.
- Diagnosticar Pre-LN vs Post-LN.

## Constrúyelo

```python
def encoder_block(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2):
    a, _ = _attn(x @ W_Q, x @ W_K, x @ W_V)
    h = layer_norm(x + a @ W_O)
    return layer_norm(h + ffn(h, W1, b1, W2, b2))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transformer-block
fase: 07
leccion: 05
---

1. Encoder: self-attn -> LN -> FFN -> LN.
2. Decoder: masked self-attn -> cross-attn -> FFN.
3. Pre-LN: x = x + sublayer(LN(x)).
4. d_ff: 4*d (GELU), 8/3*d (SwiGLU).
5. Residual + Pre-LN = estable.
```

## Ejercicios

1. **Post vs Pre-LN**: entrenar transformer pequeno
   con Post-LN y Pre-LN, comparar curva de loss.
2. **Deep network**: entrenar transformer 24
   capas, verificar que no diverge.
3. **Desafio**: implementar RoBERTa-style encoder
   stack con 12 capas.

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- "GLU Variants Improve Transformer" (Shazeer, 2020)

---

> 📚 **Adaptación al español** de la lección "[Full Transformer]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).