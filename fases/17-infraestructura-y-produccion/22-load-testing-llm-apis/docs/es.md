# Load testing LLM APIs

> Load testing: (1) Concurrent (users), (2) RPS (throughput), (3) Percentiles (p50/p95/p99), (4) Error rate (reliability), (5) Ramping (gradual+soak). LoadTest: latencies list+errors int+requests int+simulate_request (base_latency, error_rate, jitter)+percentile(p) sort+index+avg_latency+error_rate+throughput_rps(duration). ramp_load: users 1->max. Diferencias: Load = normal+expected+capacity, Stress = peak+beyond+breaking, Soak = long+hours+memory leak. Criterios: Load = normal+expected+capacity, Stress = peak+beyond+breaking, Soak = long+hours+memory, Spike = burst+sudden+auto-scale. Decision: normal -> load, peak -> stress, long -> soak, burst -> spike. Frameworks: locust, k6, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + load.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/21
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LoadTest con latencies + errors + requests.
- Implementar simulate_request.
- Implementar percentile + avg + error_rate.
- Implementar ramp_load.
- Diagnosticar load vs stress vs soak.

## Constrúyelo

```python
def simulate_request(self, base_latency=0.5, error_rate=0.01, jitter=0.1):
    self.requests += 1
    if random.random() < error_rate:
        self.errors += 1
        return None
    latency = base_latency + random.uniform(-jitter, jitter)
    self.latencies.append(latency)
    return latency
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: load-testing-llm
fase: 17
leccion: 22
---

1. LoadTest.
2. simulate + percentile.
3. ramp_load.
4. +Production.
```

## Ejercicios

1. **LoadTest**: probar
   simulate.
2. **Percentile**: probar
   p50/p99.
3. **Desafio**: integrar
   con Locust.

## Lecturas recomendadas

- "Locust" (Locust, 2024)
- "k6" (Grafana, 2024)
- "LLM Load Testing" (Locust, 2024)

---

> 📚 **Adaptación al español de la lección [Load Testing LLM APIs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).