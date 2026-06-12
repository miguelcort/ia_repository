# Cold start mitigation

> Cold start: (1) Keep warm (loaded+idle), (2) Predictive (history+pre-warm), (3) Snapshot (cache state), (4) Lazy (on-demand+defer), (5) JIT (compile cache). ModelPool: max_size+idle_timeout+loaded+warm (load if missing)+get (touch last_used)+_evict_idle (time check)+size. PredictiveWarmer: history+record_request+predict_next_models (counts+top_k)+warm_predicted (iterate). Diferencias: Warm = loaded+memory, Predictive = pre-load+history+smart, Lazy = on-demand+cheap+slow first. Criterios: Warm = predictable+high traffic+latency, Predictive = bursty+pattern+history, Lazy = rare+cheap+slow OK, Snapshot = state+heavy+recover. Decision: predict -> warm, bursty -> predictive, rare -> lazy, state -> snapshot. Frameworks: vllm, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + warm.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/09
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ModelPool con warm + get + evict.
- Implementar PredictiveWarmer con history + predict.
- Diagnosticar warm vs predictive vs lazy.
- Diagnosticar criteria.
- Diagnosticar trade-offs.

## Constrúyelo

```python
class ModelPool:
    def warm(self, model_name, load_fn):
        if model_name in self.loaded:
            return False
        if len(self.loaded) >= self.max_size:
            self._evict_idle()
        self.loaded[model_name] = {"model": load_fn(), "last_used": time.time()}
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
name: cold-start-mitigation
fase: 17
leccion: 10
---

1. ModelPool.
2. PredictiveWarmer.
3. LRU eviction.
4. +Production.
```

## Ejercicios

1. **Pool**: probar
   warm + evict.
2. **Predictive**: probar
   predict.
3. **Desafio**: integrar
   con vLLM warm pool.

## Lecturas recomendadas

- "AWS Lambda Cold Start" (AWS, 2024)
- "Predictive Warmup" (Google, 2024)
- "SnapStart" (AWS, 2024)

---

> 📚 **Adaptación al español de la lección [Cold Start Mitigation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).