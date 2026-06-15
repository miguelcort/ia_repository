# 28 — Observability con OpenTelemetry traces

> OpenTelemetry (OTel) traces para LLM apps: span per LLM call, tool call, agent step. Atributos: tokens, cost, latency, model, prompt preview. Exporters: Jaeger, Tempo, Honeycomb. Frameworks: OpenLLMetry (Traceloop), OpenInference.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/11, Fase 17
**Tiempo estimado:** ~25 minutos

## Objetivos

- OpenTelemetry instrumentation.
- Span hierarchy (agent → tool → LLM).
- Attributes (tokens, cost).
- Exporters (Jaeger, Tempo).

## El problema

OTel traces: cada operación es un span con
start_time, end_time, attributes. Hierarchy: root
span (agent run) → child span (tool call) → child
span (LLM call). Attributes: llm.model, llm.tokens,
llm.cost, llm.prompt_preview. Exporters: Jaeger
(local), Tempo (Grafana), Honeycomb (cloud). Frameworks
específicos para LLMs: OpenLLMetry (Traceloop),
OpenInference (Arize). Para agent debugging es
crítico.

## Constrúyelo

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


@tracer.start_as_current_span("llm_call")
def llm_call(prompt):
    span = trace.get_current_span()
    response = openai_call(prompt)
    span.set_attribute("llm.model", "gpt-4o")
    span.set_attribute("llm.tokens", response.usage.total_tokens)
    return response.choices[0].message.content
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-otel
fase: 19
leccion: 28
---

1. OTel setup.
2. Span hierarchy.
3. LLM attributes.
4. Exporters.
5. OpenLLMetry.
```

## Ejercicios

1. **Trace**: instrument
   5 LLM calls.
2. **Export**: Jaeger local.
3. **Desafío**: OpenLLMetry
   integration.

## Lecturas recomendadas

- "OpenTelemetry" (CNCF 2024)
- "OpenLLMetry" (Traceloop 2024)
- "OpenInference" (Arize 2024)
- "Jaeger" (Uber 2017)

---

> 📚 **Adaptación al español** de la lección
> "[28-observability-otel-traces]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
