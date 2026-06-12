# AI gateways

> AI gateways: (1) Unified (API), (2) Multi (provider), (3) Routing (by model), (4) Fallback (chain), (5) Cache (rate). Provider: name+base_url+api_key+models. AIGateway: providers+routes+cache+add_provider+add_route (alias, providers)+call (check cache+iterate providers+fallback)+list_routes. Ventajas AI gateways vs direct: Multi-provider (switch+no lock-in), Fallback (reliability), Cache (cost+latency), Rate limit (per user), Observability (logs+metrics). Criterios: Gateway = multi-provider+fallback+observability, Direct = single+cheap+dev, Custom = specific+tailored+internal. Decision: multi -> gateway, single -> direct, specific -> custom, mix -> gateway+direct. Frameworks: portkey, litellm, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + gateways.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/18
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Provider con name + base_url + models.
- Implementar AIGateway con providers + routes + cache.
- Implementar call con cache + fallback.
- Diagnosticar gateways vs direct.
- Diagnosticar criteria.

## Constrúyelo

```python
def call(self, alias, model, prompt):
    cache_key = f"{alias}:{model}:{prompt}"
    if cache_key in self.cache:
        return self.cache[cache_key], "cache"
    providers = self.routes.get(alias, [])
    for provider_name in providers:
        provider = self.providers.get(provider_name)
        if provider is None:
            continue
        if model in provider.models or not provider.models:
            response = f"[{provider.name}] response to: {prompt[:30]}"
            self.cache[cache_key] = response
            return response, provider.name
    return None, "no_provider"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: ai-gateways
fase: 17
leccion: 19
---

1. Provider + AIGateway.
2. call + fallback + cache.
3. list_routes.
4. +Production.
```

## Ejercicios

1. **Provider**: probar
   add.
2. **Gateway**: probar
   call + fallback.
3. **Desafio**: integrar
   con LiteLLM.

## Lecturas recomendadas

- "Portkey Gateway" (Portkey, 2024)
- "LiteLLM" (BerriAI, 2024)
- "OpenRouter" (OpenRouter, 2024)

---

> 📚 **Adaptación al español de la lección [AI Gateways]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).