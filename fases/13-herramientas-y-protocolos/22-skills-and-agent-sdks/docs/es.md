# Skills and agent SDKs

> Skills: reusable agent capabilities (name + description + handler + inputs/outputs). Agent SDKs: (1) Anthropic Skills (2024, skills system +reusable), (2) OpenAI Agents SDK (2024, handoffs + tools + tracing), (3) LangGraph (LangChain 2024, state machines + cycles), (4) CrewAI (2024, multi-agent + roles + tasks), (5) AutoGen (Microsoft 2024, conversational + code exec), (6) Smolagents (HF 2024, lightweight + code agents). Primitives: skills, tools (MCP + custom), memory (short + long term + persistent), state (shared + cycles), tracing (OTel + observability), handoffs (multi-agent + delegation). +Composability, +Modular, +Standardized, +Reusable, +Production. Frameworks: anthropic, openai, langchain, langgraph, crewai, autogen, hf, pydantic-ai, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, automation, multi-agent. Decision: skills -> Anthropic, tools -> OpenAI, state -> LangGraph, multi-agent -> CrewAI, code -> AutoGen, lightweight -> Smolagents, production -> combinacion. Variants: Anthropic, OpenAI, LangGraph, CrewAI, AutoGen, Smolagents, PydanticAI. 2025: +MCP + A2A + native + skills.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Skill class.
- Implementar AgentSDKBundle con add/run/list.
- Diagnosticar Anthropic vs OpenAI vs LangGraph vs CrewAI.
- Diagnosticar SDK primitives.
- Diagnosticar skills composability.

## Constrúyelo

```python
class Skill:
    def run(self, **kwargs):
        return self.handler(**kwargs)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: skills-sdks
fase: 13
leccion: 22
---

1. Skills.
2. SDKs.
3. Primitives.
4. +Composability.
5. +Production.
```

## Ejercicios

1. **Skills**: implementar
   Anthropic Skills custom.
2. **OpenAI Agents**:
   probar Agents SDK.
3. **Desafio**: custom
   agent SDK.

## Lecturas recomendadas

- "Anthropic Skills Documentation" (Anthropic, 2024)
- "OpenAI Agents SDK" (OpenAI, 2024)
- "LangGraph Documentation" (LangChain, 2024)
- "CrewAI Documentation" (CrewAI, 2024)

---

> 📚 **Adaptación al español de la lección [Skills and Agent SDKs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).