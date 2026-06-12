# Multi-region KV locality

> KV locality: (1) Route (nearest+sticky), (2) Replicate (across), (3) Sticky (same region), (4) Compliance (data residency), (5) Latency (reduce). Region: name+location+replicas+kv_cache. KVLocalityRouter: regions dict+user_region dict+add_region+assign_user+route (sticky+fallback)+get/set_cache+replicate_to. nearest_region: exact match+fallback first. Ventajas KV locality vs single: latency (near+fast), compliance (residency+GDPR), resilience (multi-region+failover), scale (per-region), locality (cache hit+reuse). Criterios: Sticky = same region+compliance+predict, Replicate = shared+cross-region+cache hit, Global = everywhere+slow+rare. Decision: sticky -> compliance, replicate -> shared, global -> rare, mix -> sticky+replicate. Frameworks: custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + locality.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/10
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Region con name + location + replicas + kv_cache.
- Implementar KVLocalityRouter con add + assign + route.
- Implementar get/set/replicate_cache.
- Implementar nearest_region.
- Diagnosticar sticky vs replicate vs global.

## Constrúyelo

```python
class KVLocalityRouter:
    def route(self, user_id):
        region_name = self.user_region.get(user_id)
        if region_name and region_name in self.regions:
            return self.regions[region_name]
        if self.regions:
            return next(iter(self.regions.values()))
        return None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: multi-region-kv-locality
fase: 17
leccion: 11
---

1. Region + KVLocalityRouter.
2. route + get/set.
3. nearest_region.
4. +Production.
```

## Ejercicios

1. **Region**: probar
   create + cache.
2. **Router**: probar
   assign + route.
3. **Desafio**: integrar
   con multi-region deploy.

## Lecturas recomendadas

- "Multi-Region LLMs" (Cloudflare, 2024)
- "GDPR Data Residency" (EU, 2024)
- "Sticky Sessions" (Nginx, 2024)

---

> 📚 **Adaptación al español de la lección [Multi-region KV Locality]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).