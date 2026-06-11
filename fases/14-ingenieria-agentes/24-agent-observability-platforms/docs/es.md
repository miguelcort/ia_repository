# Agent observability platforms

> Agent observability platforms: tracing + eval + monitoring + production. Variants: (1) Langfuse (open source +self-hosted +multi-platform +community), (2) LangSmith (managed +LangChain-native +SOTA +eval), (3) Opik (comet +LLM-native +production), (4) Phoenix (arize +open +eval), (5) Helicone, (6) Traceloop, (7) OTel, (8) custom. Features: (1) tracing (spans, attributes, events), (2) evals (quality metrics, scoring), (3) monitoring (latency, errors, cost), (4) +production (reliable, standardized). +Tracing, +Eval, +Monitoring, +Cost, +Production, +Reliable, +Standardized. Langfuse vs LangSmith: Langfuse (open +self-hosted +LLM-native +multi-platform) vs LangSmith (managed +LangChain +SOTA +eval). Frameworks: langfuse, langsmith, opik, phoenix, helicone, traceloop, opentelemetry. +Production: standard 2024-25. +Use cases: agent, observability, debugging, monitoring, cost. Decision: open -> Langfuse o Phoenix, LangChain -> LangSmith, comet -> Opik, arize -> Phoenix, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + observability.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/23
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MockObservabilityPlatform base.
- Implementar Langfuse, LangSmith, Opik, Phoenix.
- Diagnosticar features per platform.
- Diagnosticar Langfuse vs LangSmith trade-offs.

## Constrúyelo

```python
class MockObservabilityPlatform:
    def trace(self, span_name, attributes=None):
        trace = {
            "id": f"trace_{len(self.traces)}",
            "span_name": span_name,
            "attributes": attributes or {},
            "start_time": time.time(),
            "end_time": None,
        }
        self.traces.append(trace)
        return trace
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: observability-platforms
fase: 14
leccion: 24
---

1. Langfuse + LangSmith.
2. Opik + Phoenix.
3. Tracing + evals.
4. Monitoring.
5. +Production.
```

## Ejercicios

1. **Langfuse**: usar
   Langfuse self-hosted.
2. **LangSmith**: probar
   LangSmith managed.
3. **Desafio**: full
   observability stack.

## Lecturas recomendadas

- "Langfuse Documentation" (https://langfuse.com)
- "LangSmith Documentation" (https://docs.smith.langchain.com)
- "Opik by Comet" (https://www.comet.com)
- "Phoenix by Arize" (https://docs.arize.com/phoenix)

---

> 📚 **Adaptación al español de la lección [Agent Observability Platforms]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).