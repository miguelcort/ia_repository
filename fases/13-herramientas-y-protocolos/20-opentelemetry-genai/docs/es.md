# OpenTelemetry GenAI

> OpenTelemetry GenAI: tracing + metrics para LLM apps. Spans, attributes, events, semantic conventions. Standard GenAI attributes: (1) gen_ai.system (openai/anthropic/google), (2) gen_ai.request.model (gpt-4o, claude-3-5-sonnet), (3) gen_ai.usage.input_tokens (prompt tokens), (4) gen_ai.usage.output_tokens (completion tokens), (5) gen_ai.response.model (actual model). +Standardized, +Interop, +Portable, +Vendor-neutral, +Observability, +Debugging, +Production. Span structure: (1) span per LLM call (openai.chat, anthropic.messages), (2) span per tool call (tool.execute, tool.name), (3) span per agent step (agent.step, agent.think), (4) parent-child (trace_id, parent_span_id), (5) events (tool_call, tool_result, details), (6) attributes (gen_ai.*, duration). Variants: OpenTelemetry, OpenLLMetry, Traceloop, Langfuse, LangSmith, Opik, Phoenix. Frameworks: opentelemetry, langfuse, langsmith, opik, phoenix, traceloop. +Production: standard 2024-25. +Use cases: agent, RAG, automation, debugging, observability. Decision: standard -> OTel, LLM-native -> Langfuse, LangChain -> LangSmith, production -> OTel + Langfuse. 2025: +MCP + A2A + native + observability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Span class.
- Implementar Tracer.
- Implementar gen_ai_attributes (semantic conventions).
- Implementar record_tool_span.
- Diagnosticar OpenTelemetry vs Langfuse vs LangSmith.

## Constrúyelo

```python
class Span:
    def set_attribute(self, key, value):
        self.attributes[key] = value
    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attributes or {}})
    def end(self, status="ok"):
        self.end_time = time.time()
        self.status = status
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: otel-genai
fase: 13
leccion: 20
---

1. Tracing + metrics.
2. Semantic conventions.
3. gen_ai.* attributes.
4. Span per LLM/tool/step.
5. +Observability.
```

## Ejercicios

1. **OTel**: usar
   opentelemetry-instrumentation.
2. **Langfuse**: probar
   Langfuse tracing.
3. **Desafio**: LLM app
   con observability.

## Lecturas recomendadas

- "OpenTelemetry GenAI Conventions" (https://opentelemetry.io)
- "OpenLLMetry" (https://github.com/traceloop/openllmetry)
- "Langfuse Documentation" (https://langfuse.com)
- "LangSmith Documentation" (https://docs.smith.langchain.com)

---

> 📚 **Adaptación al español de la lección [OpenTelemetry GenAI]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).