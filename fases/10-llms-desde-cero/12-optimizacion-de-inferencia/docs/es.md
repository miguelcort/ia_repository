# Optimización de inferencia

> Optimizaciones SOTA: KV cache (O(n) por step, reusa pasados), Flash Attention 2/3 (O(n) memoria, 2-4x speedup), Paged Attention (vLLM, bloques no contiguos, zero waste, 2-4x throughput vs continuous), Continuous Batching (mix prefill/decode, 2-4x throughput), Prefix Caching (system prompt, 70%+ hit rate en multi-turn), Speculative Decoding (draft chico + target grande, 2-3x speedup), Tensor Parallel (multi-GPU), Chunked Prefill (long contexts en chunks), Quantization (INT4/INT8/FP8). Frameworks: vLLM, SGLang, TGI, TensorRT-LLM, llama.cpp. SOTA: Llama 70B 50+ tok/sec en A100.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/12-kv-cache-y-flash-attention, 10/11-cuantizacion
**Tiempo estimado:** ~30 minutos

## Objetivos

- Calcular KV cache memory.
- Implementar paged attention block count.
- Calcular continuous batching throughput.
- Diagnosticar prefix caching y chunked prefill.

## Constrúyelo

```python
def kv_cache_memory(n_layers, n_heads, d_k, seq_len, batch_size, num_bytes=2):
    return 2 * n_layers * n_heads * seq_len * d_k * num_bytes * batch_size
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-inference-optimization
fase: 10
leccion: 12
---

1. KV cache, Flash Attention.
2. Paged attention, continuous batching.
3. Prefix caching, chunked prefill.
4. Spec decode, tensor parallel.
5. vLLM, SGLang, TGI.
```

## Ejercicios

1. **vLLM**: deploy Llama 3 8B
   con vLLM + prefix caching.
2. **Paged**: simular paged
   attention memory savings.
3. **Desafio**: combo INT4 + spec
   decode + chunked prefill.

## Lecturas recomendadas

- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
- "vLLM: A High-Throughput and Memory-Efficient Inference Engine for LLMs" (Kwon et al., 2023)
- "SGLang: Efficient Execution of Structured Language Model Programs" (Zheng et al., 2024)
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Inference Optimization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).