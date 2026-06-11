# Self-attention desde cero

> Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. Proyecciones aprendibles, mask opcional, paralelizable, base de toda arquitectura transformer moderna.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/01-por-que-transformers
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar softmax numericamente estable.
- Implementar scaled dot-product attention.
- Aplicar causal mask para decoder.
- Diagnosticar varianza de attention.

## Constrúyelo

```python
def self_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask  # mask: -inf donde bloqueas
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-self-attention
fase: 07
leccion: 02
---

1. Formula: softmax(QK^T / sqrt(d_k)) V.
2. Proyecciones: Q, K, V de la misma X (self) o X y
   context (cross).
3. Mask: causal en decoder, padding en encoder, ambos en
   seq2seq.
4. Scale: sqrt(d_k) para estabilidad.
5. Output: weighted sum de V.
```

## Ejercicios

1. **Perfilado**: medir tiempo y memoria de self-attention
   para n=512, 1024, 2048, 4096.
2. **Variantes**: implementar additive attention (Bahdanau)
   y comparar con dot-product.
3. **Desafio**: implementar self-attention con Flash
   Attention tiling (block-wise softmax).

## Lecturas recomendadas

- "Attention Is All You Need" (Vaswani et al., 2017)
- "FlashAttention" (Dao et al., 2022)
- "Self-Attention with Relative Position Representations" (Shaw et al., 2018)

---

> 📚 **Adaptación al español** de la lección "[Self-Attention From Scratch]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).