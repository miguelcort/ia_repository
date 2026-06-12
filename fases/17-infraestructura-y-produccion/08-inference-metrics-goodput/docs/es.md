# Inference metrics goodput

> Metricas: (1) Goodput (SLO+fraction), (2) TTFT (first token+latency), (3) TPOT (per token+throughput), (4) ITL (inter-token), (5) E2E (end-to-end+total). InferenceMetrics: record_request(ttft, tpot, tokens)+avg_ttft() mean+avg_tpot() mean+tokens_per_second (tokens/duration)+requests_per_second (count/duration)+goodput(slo_ttft, slo_tpot) fraction meeting SLO+e2e_latency_p99 (ttft + tpot*tokens + p99). Diferencias: Latency = time per req, Throughput = reqs/sec, Goodput = reqs meeting SLO. Criterios: p50 = typical+median, p99 = tail+worst, avg = mean+overall, Goodput = SLO+quality. Decision: typical -> p50, tail -> p99, mean -> avg, quality -> goodput. Frameworks: custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + metrics.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar InferenceMetrics con record_request.
- Implementar avg_ttft + avg_tpot.
- Implementar goodput con SLO check.
- Implementar e2e_latency_p99.
- Diagnosticar metrics types.

## Constrúyelo

```python
def goodput(self, slo_ttft, slo_tpot):
    if not self.requests:
        return 0.0
    good = sum(1 for r in self.requests if r["ttft"] <= slo_ttft and r["tpot"] <= slo_tpot)
    return good / len(self.requests)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: inference-metrics-goodput
fase: 17
leccion: 08
---

1. InferenceMetrics.
2. record + avg.
3. goodput.
4. e2e_p99.
5. +Production.
```

## Ejercicios

1. **Metrics**: probar
   record + avg.
2. **Goodput**: probar
   SLO.
3. **Desafio**: integrar
   con tu monitoring.

## Lecturas recomendadas

- "LLM Inference Metrics" (Anyscale, 2024)
- "Goodput" (Spectro Cloud, 2024)
- "OpenTelemetry LLM" (CNCF, 2024)

---

> 📚 **Adaptación al español de la lección [Inference Metrics Goodput]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).