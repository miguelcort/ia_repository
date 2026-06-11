# Failure modes agentic

> Failure modes en agents: (1) tool_error (tool call failed, high), (2) hallucination (LLM hallucinated, high), (3) infinite_loop (agent looped, critical), (4) context_overflow (context exceeded, high), (5) security (prompt injection, tool poisoning, critical), (6) timeout (medium), (7) rate_limit (medium). +Robust, +Resilient, +Reliable, +Production. Defenses: tool error (retry + fallback), hallucination (debate + self-refine + CRITIC), infinite loop (max_iter + timeout + budget), context overflow (MemGPT + summarize + truncate + compression), security (validation + provenance + sandboxing). Variants: retry, fallback, debate, self-refine, CRITIC, max_iter, timeout, budget, MemGPT, summarize, truncate, compression, validation, provenance. Frameworks: langchain, autogen, smolagents, langgraph, memgpt, letta, openai, anthropic. +Production: standard 2024-25. +Use cases: agent, debugging, monitoring, production, long context. Decision: tool -> retry+fallback, hallucination -> debate+self-refine, loop -> max_iter, overflow -> MemGPT, security -> validation, production -> defense in depth. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + defenses.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar FailureMode class.
- Implementar AgentFailureMonitor con modes dict.
- Implementar record_failure + get_report + total.
- Diagnosticar defenses por failure mode.

## Constrúyelo

```python
class FailureMode:
    def __init__(self, name, description, severity="medium"):
        self.name = name
        self.description = description
        self.severity = severity
        self.count = 0

    def record(self):
        self.count += 1
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: failure-modes
fase: 14
leccion: 26
---

1. Tool errors.
2. Hallucinations.
3. Infinite loops.
4. Context overflow.
5. +Robust.
```

## Ejercicios

1. **Failure monitor**:
   implementar monitor custom.
2. **Defense in depth**:
   agregar retries + fallbacks.
3. **Desafio**: production
   failure monitor.

## Lecturas recomendadas

- "Failure Modes in Agentic Systems" (Anthropic, 2024)
- "MemGPT for Context Overflow" (Packer, 2023)
- "Self-Refine for Hallucinations" (Madaan, 2023)
- "Multi-Agent Debate for Factuality" (Du, 2023)

---

> 📚 **Adaptación al español de la lección [Failure Modes Agentic]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).