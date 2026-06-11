# OTel GenAI conventions

> OTel GenAI semantic conventions para agents. Attributes: gen_ai.agent.name (agent identifier), gen_ai.span.kind = 'agent', gen_ai.request.model (model), gen_ai.agent.input (input text), gen_ai.agent.output (output text), gen_ai.agent.steps (number of steps). Events: (1) agent_start, (2) agent_end, (3) tool_call (tool.name + tool.arguments + tool.result + tool.latency_ms), (4) handoff (from_agent + to_agent + reason). +Multi-agent (handoffs), +Tool use, +Memory, +Observability, +Standardized, +Interop, +Portable, +Debugging, +Trace, +Repro, +Reliable, +Production. Variants: OTel GenAI, Langfuse, LangSmith, Opik, Phoenix, Helicone, Traceloop, custom. Frameworks: opentelemetry, langfuse, langsmith, opik, phoenix, traceloop. +Production: standard 2024-25. +Use cases: agent, multi-agent, observability, debugging, handoffs. Decision: standard -> OTel, LLM-native -> Langfuse, LangChain -> LangSmith, production -> OTel + Langfuse. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + observability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/20, 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar AgentSpan con attributes + events.
- Implementar record_agent_run.
- Implementar record_tool_call.
- Implementar record_handoff.
- Diagnosticar OTel vs Langfuse vs LangSmith.

## Constrúyelo

```python
class AgentSpan:
    def set_attribute(self, key, value):
        self.attributes[key] = value
    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "timestamp": time.time(), "attributes": attributes or {}})
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
fase: 14
leccion: 23
---

1. AgentSpan.
2. gen_ai.agent.*.
3. Tool call events.
4. Handoff events.
5. +Observability.
```

## Ejercicios

1. **OTel**: usar
   opentelemetry-instrumentation.
2. **Langfuse**: probar
   Langfuse tracing.
3. **Desafio**: agent
   con observability.

## Lecturas recomendadas

- "OpenTelemetry GenAI Conventions" (https://opentelemetry.io)
- "OpenLLMetry" (https://github.com/traceloop/openllmetry)
- "Langfuse Documentation" (https://langfuse.com)
- "Agent Observability" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [OTel GenAI Conventions]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).