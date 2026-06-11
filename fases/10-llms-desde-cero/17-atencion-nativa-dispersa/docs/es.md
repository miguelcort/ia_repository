# Atención nativa dispersa (NSA)

> Native Sparse Attention (NSA, DeepSeek 2025): 3 branches — compression (block-wise K, V compression), selection (top-k important blocks por importance score), sliding window (local attention). Hardware-efficient via Triton kernels, custom CUDA. 12x speedup en long context (64K-128K) vs full attention. +Quality en long context benchmarks. DeepSeek-V3 (671B MoE) usa NSA + MLA + MoE = SOTA open source. Memory O(n·compressed) vs O(n²). Compatible con GQA, MLA, MoE. Hoy: NSA + MLA en DeepSeek-V3 son SOTA para long context.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/03-multi-head-attention, 10/12-optimizacion-de-inferencia
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar dense attention.
- Implementar sliding window attention.
- Implementar compressed attention.
- Comparar memory complexity.
- Diagnosticar NSA architecture.

## Constrúyelo

```python
def compressed_attention(Q, K, V, block_size=4):
    n_blocks = K.shape[0] // block_size
    K_compressed = K[:n_blocks*block_size].reshape(n_blocks, block_size, -1).mean(axis=1)
    V_compressed = V[:n_blocks*block_size].reshape(n_blocks, block_size, -1).mean(axis=1)
    scores = Q @ K_compressed.T / np.sqrt(Q.shape[-1])
    return softmax(scores) @ V_compressed
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-nsa
fase: 10
leccion: 17
---

1. 3 branches: compression, selection, sliding.
2. 12x speedup.
3. Triton kernels.
4. DeepSeek-V3: NSA + MLA + MoE.
5. Long context SOTA.
```

## Ejercicios

1. **NSA**: implementar NSA
   simplificado.
2. **Memory**: comparar
   dense vs compressed.
3. **Desafio**: NSA + MoE
   end-to-end.

## Lecturas recomendadas

- "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (DeepSeek-AI, 2025)
- "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (DeepSeek-AI, 2024)
- "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2025)
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

---

> 📚 **Adaptación al español** de la lección "[Native Sparse Attention]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).