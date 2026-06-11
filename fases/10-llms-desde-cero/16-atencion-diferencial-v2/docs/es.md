# Atención diferencial v2

> Differential Attention (Microsoft 2024, Tianzhu Ye): DiffAttn = (Q₁·K₁ᵀ - λ·Q₂·K₂ᵀ) / √(2d). Splits Q, K en dos mitades, substrae noise component. Phi-4 (Microsoft 2024-25) usa Diff Attention: 14B params, SOTA en coding (HumanEval 91%, MATH 80%). Beneficios: -50% KV cache memory, +5-10% en coding/math, free "anti-noise". Compatible con Flash Attention, RoPE, GQA. Drop-in replacement. Trade-off: small models no se benefician tanto. Phi-4 supera Llama 3.1 8B en coding y math per-param.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/02-self-attention-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar standard attention.
- Implementar differential attention.
- Diagnosticar split y scale.
- Comparar Phi-4 vs Llama 3.

## Constrúyelo

```python
def differential_attention(Q, K, V, lambda_q=1.0, mask=None):
    d_k = Q.shape[-1]
    half = d_k // 2
    Q1, Q2 = Q[:, :half], Q[:, half:]
    K1, K2 = K[:, :half], K[:, half:]
    s1 = Q1 @ K1.T / np.sqrt(d_k)
    s2 = Q2 @ K2.T / np.sqrt(d_k)
    diff = s1 - lambda_q * s2
    if mask is not None:
        diff = diff + mask
    weights = softmax(diff, axis=-1)
    return weights @ V
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-diff-attention
fase: 10
leccion: 16
---

1. DiffAttn = Q1@K1^T - lambda*Q2@K2^T.
2. Split Q, K en mitades.
3. -50% KV memory.
4. Phi-4 91% HumanEval.
5. Coding, math SOTA.
```

## Ejercicios

1. **Diff Attention**: implementar
   diff attention from scratch.
2. **Phi-4**: comparar Phi-4 14B
   con Llama 3.1 8B.
3. **Desafio**: implementar
   Diff + GQA combinados.

## Lecturas recomendadas

- "Differential Transformer" (Tianzhu Ye, Microsoft, 2024)
- "Phi-4 Technical Report" (Microsoft, 2024)
- "Multi-Latent Attention (MLA)" (DeepSeek-AI, 2024)

---

> 📚 **Adaptación al español** de la lección "[Differential Attention V2]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).