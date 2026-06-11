# Prompt caching

> Prompt caching: provider cache (Anthropic 1.25x writes, 0.1x reads, cache_control breakpoints; OpenAI -50% automatic cached; Gemini implicit prefix cache), semantic cache (GPTCache, LangCache, vector store, threshold 0.85-0.95, +Recall), response cache (Redis, memcached, Upstash, Vercel KV, key=hash(prompt+model+temp)). TTL: 1h response, 1d semantic, 5min provider. Invalidation: TTL + manual flush (CLI, admin UI) + event-based (Pub/sub Redis/Kafka) + version-based. Monitoring: hit rate, miss rate, latency, cost (Helicone, Portkey, LangSmith, Prometheus). Multi-tier: provider + semantic + response. Frameworks: GPTCache (Python, 1.0+), LangCache, LangChain caching, LlamaIndex caching. Production: 30-60% hit rate, -50-90% cost, -5-10x latency. SOTA 2024-25: multi-tier + monitoring + invalidation. Frontier: +reasoning cache (R1-style), +multimodal cache (vision, audio), +agentic cache.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/11-caching-y-costo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar cache_key_from_prompt.
- Implementar should_cache heuristic.
- Implementar cache_hit_rate.
- Implementar TTL por cache type.
- Diagnosticar invalidation strategies.

## Constrúyelo

```python
def cache_key_from_prompt(prompt, model, temperature):
    h = hashlib.md5(f"{model}:{temperature}:{prompt}".encode()).hexdigest()
    return h
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-caching
fase: 11
leccion: 15
---

1. Provider, semantic, response.
2. TTL, invalidation.
3. Hit rate monitoring.
4. Multi-tier cache.
5. GPTCache, LangCache.
```

## Ejercicios

1. **Cache**: implementar cache
   multi-tier.
2. **Hit rate**: medir hit rate
   en production.
3. **Desafio**: event-based
   invalidation con Pub/sub.

## Lecturas recomendadas

- "GPTCache: Efficient Semantic Cache for LLM" (2023)
- "Anthropic Prompt Caching" (Anthropic, 2024)
- "Redis as a LLM Cache" (Redis Labs, 2024)
- "Semantic Caching for LLMs" (LangCache, 2024)

---

> 📚 **Adaptación al español de la lección [Prompt Caching]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).