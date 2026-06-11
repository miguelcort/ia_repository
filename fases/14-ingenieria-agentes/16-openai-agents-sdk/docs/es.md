# OpenAI Agents SDK

> OpenAI Agents SDK (2024): production agent framework built on Chat Completions. Components: (1) Agent (name + instructions + tools + handoffs list + model), (2) Runner (agent + tracing + max_turns, run loop), (3) Guardrail (name + check_fn, validate text), (4) handoffs (transfer to another agent via keyword match). +Handoffs, +Tools, +Tracing, +Guardrails, +Multi-agent, +Routing, +Specialization, +Reliable, +Scalable, +Production. Variants: OpenAI Agents seminal, Anthropic Skills (2024 +skills +reusable +Claude), LangGraph (+stateful +cycles), smolagents (HF 2024 +code +lightweight), CrewAI, AutoGen. Frameworks: openai, anthropic, langchain, smolagents, crewai, autogen. +Production: standard 2024-25. +Use cases: agent, multi-agent, handoffs, skills, triage, guardrails. Decision: handoffs -> OpenAI, skills -> Anthropic, cycles -> LangGraph, code -> smolagents, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13, 14/14, 14/15
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Agent con handoffs.
- Implementar Runner con max_turns.
- Implementar Guardrail.
- Diagnosticar OpenAI Agents vs Anthropic Skills.
- Diagnosticar handoff flow.

## Constrúyelo

```python
class Agent:
    def run(self, input_text, session=None):
        if self.handoffs and "transfer" in input_text.lower():
            target = self.handoffs[0]
            self.handoff_count += 1
            return {"handoff_to": target.name, "agent": self.name}
        return {"response": f"[{self.name}] processed", "agent": self.name}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: openai-agents
fase: 14
leccion: 16
---

1. Handoffs.
2. Tools.
3. Tracing.
4. Guardrails.
5. +Production.
```

## Ejercicios

1. **OpenAI Agents**: usar
   OpenAI Agents con handoffs.
2. **Tracing**: agregar
   OpenTelemetry tracing.
3. **Desafio**: production
   multi-agent con handoffs.

## Lecturas recomendadas

- "OpenAI Agents SDK Documentation" (OpenAI, 2024)
- "Building Agents with OpenAI" (OpenAI, 2024)
- "Handoffs and Tracing" (OpenAI, 2024)
- "OpenAI Swarm" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [OpenAI Agents SDK]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).