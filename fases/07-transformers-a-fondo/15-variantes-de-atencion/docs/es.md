# Variantes de atención

> Variantes de attention: standard O(n²), linear (kernel feature map O(n)), sliding window (local O(n·w), Mistral 4096), global-local (Longformer: sliding + global tokens), sparse (BigBird: sliding + global + random), state-space (Mamba/RWKV O(n) sin attention). Trade-offs: quality, memoria, compute, complejidad. Standard + sliding + RoPE + Flash = sweet spot para LLMs.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/12-kv-cache-y-flash-attention
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar linear attention con kernel feature map.
- Implementar sliding window attention.
- Implementar global-local attention.
- Diagnosticar cuándo usar cada variante.

## Constrúyelo

```python
def linear_attention(Q, K, V, kernel="elu", eps=1e-6):
    phi = lambda x: np.maximum(x, 0) + 1 if kernel == "elu" else np.maximum(x, 0)
    Q_p, K_p = phi(Q), phi(K)
    KV = K_p.T @ V  # (d, d)
    out = (Q_p @ KV) / (Q_p @ K_p.sum(axis=0).reshape(-1, 1) + eps)
    return out, None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-attention-variants
fase: 07
leccion: 15
---

1. Standard: O(n^2).
2. Linear: kernel, O(n).
3. Sliding: O(n*w). Mistral w=4096.
4. Global-local: Longformer.
5. SSM: Mamba, RWKV.
```

## Ejercicios

1. **Comparar**: standard vs linear vs sliding
   en small LM. Medir quality y memoria.
2. **Longformer**: implementar y evaluar en
   long-doc classification.
3. **Desafio**: implementar Mamba SSM
   (selective state-space).

## Lecturas recomendadas

- "Longformer: The Long-Document Transformer" (Beltagy et al., 2020)
- "Big Bird: Transformers for Longer Sequences" (Zaheer et al., 2020)
- "Linear Attention Transformer" (Katharopoulos et al., 2020)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

---

> 📚 **Adaptación al español** de la lección "[Attention Variants]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).