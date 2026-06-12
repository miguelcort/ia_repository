# LLM observability

> Observability: (1) Traces (span+tree), (2) Metrics (counters+histograms), (3) Logs (structured), (4) OpenTelemetry (standard), (5) Cost (track per request). Span: id+name+parent_id+start+end+attributes+events+set_attribute+add_event+finish+duration. Tracer: spans dict+start_span(name, parent_id)+get_trace(root_id). Metrics: counters dict+histograms dict+increment(name, value)+observe(name, value)+get_counter+get_histogram_stats. Ventajas OTel vs custom: Standard (OTel spec+W3C trace), Vendor-neutral (Jaeger+Datadog+Honeycomb), Auto-instrumentation (HTTP+DB), Distributed (cross-service), Cost exporters (per backend). Criterios: OTel = standard+multi-vendor+future-proof, Custom = simple+cheap+single use, Vendor = deep+managed+lock-in. Decision: standard -> OTel, simple -> custom, deep -> vendor, mix -> OTel+vendor. Frameworks: otel, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + observability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/12
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Span con id + name + parent + attributes + events.
- Implementar Tracer con start_span + get_trace.
- Implementar Metrics con counters + histograms.
- Diagnosticar OTel advantages.
- Diagnosticar criteria.

## Constrúyelo

```python
class Span:
    def finish(self):
        self.end = time.time()

    def duration(self):
        if self.end is None:
            return time.time() - self.start
        return self.end - self.start
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llm-observability
fase: 17
leccion: 13
---

1. Span + Tracer.
2. Metrics.
3. Distributed.
4. +Production.
```

## Ejercicios

1. **Span**: probar
   attribute + event.
2. **Tracer**: probar
   get_trace.
3. **Desafio**: integrar
   con OTel SDK.

## Lecturas recomendadas

- "OpenTelemetry" (CNCF, 2024)
- "Distributed Tracing" (Sigelman, 2010)
- "LLM Observability" (Langfuse, 2024)

---

> 📚 **Adaptación al español de la lección [LLM Observability]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).