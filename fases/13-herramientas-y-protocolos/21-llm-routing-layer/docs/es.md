# LLM routing layer

> LLM routing: intelligent routing entre modelos. Strategies: (1) cost (cheapest model), (2) latency (fastest avg), (3) capability (matches task), (4) load balancing (round-robin, weighted), (5) fallback chain (try primary on error -> next, +reliable +uptime +resilient), (6) semantic (embedding similarity con model descriptions, +accurate +smart), (7) hybrid (combina strategies). +Cost-aware, +Reliable, +Flexible, +Production. Frameworks: litellm, portkey, openrouter, semantic-router, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation, multi-model, cost optimization. Decision: cost -> cost, speed -> latency, special -> capability, scale -> LB, production -> fallback + hybrid. Variants: cost, latency, capability, LB, fallback, semantic, hybrid. Trade-offs: cada uno + specialty, routing + flexible, single + simple. 2025: +MCP + A2A + native + routing.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/09
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar route_by_cost, route_by_latency, route_by_capability.
- Implementar fallback_chain.
- Implementar load_balance_round_robin.
- Implementar semantic_route.
- Diagnosticar routing strategies.

## Constrúyelo

```python
def route_by_cost(prompt, models):
    sorted_models = sorted(models, key=lambda m: m["cost_per_1k"])
    return sorted_models[0]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llm-routing
fase: 13
leccion: 21
---

1. Cost-based.
2. Latency-based.
3. Capability.
4. Fallback chain.
5. +Production.
```

## Ejercicios

1. **LiteLLM**: usar
   litellm con routing.
2. **Portkey**: probar
   portkey fallback.
3. **Desafio**: custom
   routing con semantic.

## Lecturas recomendadas

- "LiteLLM Documentation" (https://docs.litellm.ai)
- "Portkey Documentation" (https://portkey.ai/docs)
- "OpenRouter" (https://openrouter.ai)
- "Semantic Router" (https://github.com/aurelio-ai/semantic-router)

---

> 📚 **Adaptación al español de la lección [LLM Routing Layer]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).