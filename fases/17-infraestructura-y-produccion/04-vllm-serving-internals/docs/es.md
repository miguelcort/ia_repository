# vLLM serving internals

> vLLM internals: (1) PagedAttention (virtual mem+KV cache), (2) Continuous (batching+throughput), (3) Prefix (cache+reuse), (4) Tensor parallelism (multi-GPU+sharding), (5) High throughput. PagedAttention: block_size+num_blocks+blocks+allocate (check free+assign)+free+memory_used/total. ContinuousBatcher: max_batch+add (check)+step (increment+filter done). prefix_cache_hit: longest startswith. Ventajas PagedAttention: memory (efficient+no waste), less fragmentation (block-level+allocator), higher throughput (continuous+concurrent), longer context (paged+scalable), multi-tenant (isolated+shared). Criterios: vLLM = throughput+PagedAttn+OSS, TGI = HF+simple+multi-model, TRT-LLM = NVIDIA+perf+compile, SGLang = structured+radix+lang. Decision: throughput -> vLLM, simple -> TGI, NVIDIA -> TRT-LLM, structured -> SGLang. Frameworks: vllm, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + serving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PagedAttention con blocks.
- Implementar ContinuousBatcher.
- Implementar prefix_cache_hit.
- Diagnosticar PagedAttention advantages.
- Diagnosticar vLLM vs TGI vs TRT-LLM.

## Constrúyelo

```python
class PagedAttention:
    def allocate(self, seq_id, num_tokens):
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        free = [i for i, b in enumerate(self.blocks) if b is None]
        if len(free) < blocks_needed:
            return False
        allocated = free[:blocks_needed]
        for idx in allocated:
            self.blocks[idx] = (seq_id, idx)
        self.allocations[seq_id] = allocated
        return True
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: vllm-serving-internals
fase: 17
leccion: 04
---

1. PagedAttention.
2. ContinuousBatcher.
3. prefix_cache_hit.
4. +Production.
```

## Ejercicios

1. **PagedAttention**: probar
   allocate.
2. **ContinuousBatcher**: probar
   step.
3. **Desafio**: integrar
   con vLLM real.

## Lecturas recomendadas

- "vLLM" (Kwon, 2023)
- "PagedAttention" (Kwon, 2023)
- "Continuous Batching" (Yu, 2022)

---

> 📚 **Adaptación al español de la lección [vLLM Serving Internals]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).