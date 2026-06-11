# Caching y costo

> Caching: provider prompt cache (Anthropic 1.25x writes, 0.1x reads; OpenAI -50%; Gemini implicit), semantic cache (GPTCache, LangCache, vector store, threshold 0.85-0.95, +Recall vs exact), response cache (Redis, memcached, in-memory), KV cache (Flash Attention, paged attention, prefix cache). Cost monitoring: per request, per user, per model, daily/monthly, anomaly detection, budgets + alerts. Tools: Helicone (LLM observability), Portkey (LLM gateway), LangSmith, OpenLLMetry, CloudZero. Beneficios: -50-90% cost, -5-10x latency. Frameworks: LangChain caching, LlamaIndex caching. SOTA 2024-25: provider cache + semantic cache + KV cache + monitoring. Frontier: reasoning cache (R1-style), multimodal cache, agentic cache.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/06-rag, 11/08-fine-tuning-lora
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar cost estimator.
- Implementar response cache mock.
- Implementar prompt cache savings.
- Diagnosticar semantic cache y provider cache.

## Constrúyelo

```python
def estimate_cost(n_in, n_out, cost_in, cost_out):
    return (n_in / 1000) * cost_in + (n_out / 1000) * cost_out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: caching-cost
fase: 11
leccion: 11
---

1. Provider prompt cache.
2. Semantic cache.
3. Response cache (Redis).
4. KV cache (inference).
5. Helicone, Portkey.
```

## Ejercicios

1. **Cost**: implementar cost
   calculator por user.
2. **Cache**: implementar
   semantic cache.
3. **Desafio**: provider prompt
   cache + monitoring.

## Lecturas recomendadas

- "Prompt Caching" (Anthropic, 2024)
- "GPTCache: A Semantic Cache for LLMs" (2023)
- "Helicone: LLM Observability" (2024)
- "Portkey: LLM Gateway" (2024)

---

> 📚 **Adaptación al español** de la lección "[Caching Cost]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).