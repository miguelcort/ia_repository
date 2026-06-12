# Model routing

> Model routing: (1) Route by (complexity+cost+latency), (2) Capability (max), (3) Cascade (cheap first+escalate), (4) Specialist (per domain), (5) Fall-back (backup). ModelRoute: cost_per_1k+max_complexity+avg_latency. ModelRouter: routes+strategy cheapest/fastest/best+route (filter+min/max)+estimate_cost. detect_complexity: length (len/100)+keywords (analyze/complex/code/math/research). Diferencias: Cascade = cheap->expensive+escalate+adaptive, Specialist = per-domain+custom+specialized, Router = by criteria+static+configurable. Criterios: Cheapest = cost+bulk+predict, Fastest = latency+real-time+UI, Best = quality+hard tasks+expensive, Cascade = adaptive+cheap first+escalate. Decision: cost -> cheapest, latency -> fastest, quality -> best, adaptive -> cascade. Frameworks: langchain, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + routing.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/15
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ModelRoute con cost + max_complexity + latency.
- Implementar ModelRouter con strategy.
- Implementar route cheapest/fastest/best.
- Implementar detect_complexity.
- Diagnosticar cascade vs specialist vs router.

## Constrúyelo

```python
class ModelRouter:
    def route(self, complexity, estimated_tokens=1000):
        candidates = [r for r in self.routes if r.max_complexity >= complexity]
        if not candidates:
            return None
        if self.strategy == "cheapest":
            return min(candidates, key=lambda r: r.cost_per_1k)
        if self.strategy == "fastest":
            return min(candidates, key=lambda r: r.avg_latency)
        if self.strategy == "best":
            return max(candidates, key=lambda r: r.max_complexity)
        return candidates[0]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: model-routing
fase: 17
leccion: 16
---

1. ModelRoute + ModelRouter.
2. strategy.
3. detect_complexity.
4. +Production.
```

## Ejercicios

1. **Router**: probar
   strategies.
2. **Complexity**: probar
   keywords.
3. **Desafio**: implementar
   cascade.

## Lecturas recomendadas

- "Model Cascading" (Yue, 2024)
- "LLM Routing" (Ding, 2024)
- "RouteLLM" (Ong, 2024)

---

> 📚 **Adaptación al español de la lección [Model Routing]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).