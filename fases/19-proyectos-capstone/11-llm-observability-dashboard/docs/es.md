# 11 — LLM observability dashboard

> LLM observability: traces (Langfuse, LangSmith, Arize Phoenix), metrics (latency, cost, tokens), eval (online + offline), guardrails (guardrails-ai), drift detection. Standard en producción LLM apps.

**Tipo:** Capstone
**Lenguajes:** Python (backend), TypeScript (UI)
**Prerrequisitos:** Fase 11 (LLM), Fase 13 (tools), Fase 17
**Tiempo estimado:** 20 horas

## Objetivos

- Implementar OpenTelemetry tracing.
- Métricas: latency, tokens, cost.
- Online eval (LLM-as-judge).
- Drift detection + alerts.

## El problema

LLM observability (Langfuse, LangSmith, Arize
Phoenix, Honeycomb) es critical en producción.
Captura: (1) Traces: cada call, prompt, response,
latency, cost. (2) Metrics: p50/p99 latency, tokens,
cost, error rate. (3) Online eval: LLM-as-judge sobre
samples. (4) Drift detection: PSI, KS-test sobre
embeddings. (5) Alerts: latency spike, quality drop,
cost over budget. OpenTelemetry es el standard.

## Constrúyelo

```python
from langfuse import Langfuse
from opentelemetry import trace


langfuse = Langfuse(public_key="...", secret_key="...")


@trace.instrument
def llm_call(prompt):
    span = trace.get_current_span()
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    span.set_attribute("llm.tokens", response.usage.total_tokens)
    span.set_attribute("llm.cost", calculate_cost(response.usage))
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
name: prompt-llm-observability
fase: 19
leccion: 11
---

1. OpenTelemetry traces.
2. Token + cost tracking.
3. Online eval (LLM judge).
4. Drift detection.
5. Alerting.
```

## Ejercicios

1. **Langfuse**: integrar en app.
2. **Metrics**: dashboards
   Grafana.
3. **Desafío**: drift detection
   + auto-rollback.

## Lecturas recomendadas

- "Langfuse" (2023)
- "OpenTelemetry GenAI" (2024)
- "Arize Phoenix" (2024)
- "Honeycomb LLM Observability"

---

> 📚 **Adaptación al español** de la lección
> "[11-llm-observability-dashboard]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
