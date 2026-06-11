# KV cache y Flash Attention

> KV cache: reusa K, V pasados en inference autoregresiva. Memoria O(n_layers · seq · n_kv · d_k). Reducciones: MQA, GQA, sliding window, paged attention, quantization. Flash Attention: compute en bloques, online softmax, memoria O(n) en vez de O(n²). 2-4x speedup training. Flash 2/3 (Hopper, async, fp8). Habilita context 128K+ en producción.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/02-self-attention-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar KV cache por capa y head.
- Implementar Flash Attention con online softmax.
- Comparar memoria naive vs flash.
- Diagnosticar MQA/GQA/sliding window.

## Constrúyelo

```python
def flash_attention_demo(Q, K, V, block_size=64):
    O = np.zeros_like(Q)
    m_i = -np.inf * np.ones(Q.shape[0])
    l_i = np.zeros(Q.shape[0])
    for start in range(0, Q.shape[0], block_size):
        Kb = K[start:end]
        Vb = V[start:end]
        S = Q @ Kb.T / np.sqrt(d_k)
        m_new = np.maximum(m_i, S.max(axis=-1))
        P = np.exp(S - m_new.reshape(-1, 1))
        l_new = l_i * np.exp(m_i - m_new) + P.sum(axis=-1)
        O = (l_i * np.exp(m_i - m_new)).reshape(-1, 1) * O + P @ Vb
        O = O / l_new.reshape(-1, 1)
        m_i, l_i = m_new, l_new
    return O
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-kv-cache-flash
fase: 07
leccion: 12
---

1. KV cache: O(n) por step.
2. Flash Attention: O(n) memoria.
3. Online softmax para tiling.
4. MQA/GQA: reducir cache.
5. Paged attention (vLLM).
```

## Ejercicios

1. **GQA**: implementar Grouped-Query Attention
   con 8 grupos.
2. **Sliding window**: implementar attention
   con ventana fija.
3. **Desafio**: implementar paged attention
   con bloques de tamaño variable.

## Lecturas recomendadas

- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[KV Cache Flash Attention]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).