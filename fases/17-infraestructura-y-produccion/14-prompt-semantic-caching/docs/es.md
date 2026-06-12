# Prompt semantic caching

> Semantic cache: (1) Embeddings (vector+encode), (2) Similarity (cosine+match), (3) Threshold (min sim), (4) Hit rate (cache hits), (5) Cost (reduce+latency). cosine_similarity: dot/norms+0 if zero. SemanticCache: threshold+entries+add(embedding, response)+lookup (iterate+max sim >= threshold)+size+hit_rate. Diferencias: Exact = literal+cheap+hash+no false positive, Semantic = similar+embeddings+smart+tolerance. Criterios: Exact = repeat+cheap+hash, Semantic = variation+smart+embeddings, Prefix = shared+LLM+KV, None = cold+no benefit. Decision: repeat -> exact, variation -> semantic, shared -> prefix, cold -> none. Frameworks: gptcache, redis, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + caching.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar cosine_similarity.
- Implementar SemanticCache con entries.
- Implementar add + lookup con threshold.
- Implementar hit_rate.
- Diagnosticar exact vs semantic.

## Constrúyelo

```python
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-semantic-caching
fase: 17
leccion: 14
---

1. cosine_similarity.
2. SemanticCache.
3. add + lookup.
4. hit_rate.
5. +Production.
```

## Ejercicios

1. **Cosine**: probar
   identical + opposite.
2. **Cache**: probar
   threshold.
3. **Desafio**: integrar
   con Redis + embeddings.

## Lecturas recomendadas

- "GPTCache" (Zilliz, 2024)
- "Semantic Cache" (LangChain, 2024)
- "Redis Vector" (Redis, 2024)

---

> 📚 **Adaptación al español de la lección [Prompt Semantic Caching]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).