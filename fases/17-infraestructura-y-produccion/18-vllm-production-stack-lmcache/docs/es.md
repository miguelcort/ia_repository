# vLLM production stack LMCache

> LMCache: (1) Distributed (KV), (2) Prefix (sharing), (3) GPU (memory tier), (4) Eviction (LRU), (5) vLLM (stack). LMCache: max_size_mb+entries+access_log+put (check size+evict if needed)+get (touch last_access)+_evict LRU sort+size_mb() sum+count()+hit_rate(). Ventajas LMCache vs local: Distributed (multi-node+shared), Prefix sharing (reuse), Multi-instance (across pods), Eviction (LRU+memory), Scale (tier+GPU/SSD/HDD). Criterios: LMCache = multi-instance+distributed+tier, Local = single+cheap+dev, Redis = shared+generic+KV, vLLM = simple+native+small. Decision: multi -> LMCache, single -> local, shared -> Redis, simple -> vLLM. Frameworks: vllm, lmcache, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + lmcache.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/17
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LMCache con max_size_mb.
- Implementar put + get + _evict.
- Implementar size_mb + count + hit_rate.
- Diagnosticar advantages.
- Diagnosticar criteria.

## Constrúyelo

```python
def _evict(self, needed_mb):
    sorted_entries = sorted(self.entries.items(), key=lambda kv: kv[1]["last_access"])
    for k, v in sorted_entries:
        if self.size_mb() + needed_mb <= self.max_size_mb:
            return
        del self.entries[k]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: vllm-lmcache
fase: 17
leccion: 18
---

1. LMCache.
2. put + get + evict.
3. hit_rate.
4. +Production.
```

## Ejercicios

1. **LMCache**: probar
   put + evict.
2. **Hit rate**: probar
   statistics.
3. **Desafio**: integrar
   con vLLM.

## Lecturas recomendadas

- "LMCache" (LMCache, 2024)
- "vLLM Stack" (vLLM, 2024)
- "Distributed KV" (Zheng, 2024)

---

> 📚 **Adaptación al español de la lección [vLLM Production Stack LMCache]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).