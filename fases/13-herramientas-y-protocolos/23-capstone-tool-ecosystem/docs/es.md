# Capstone tool ecosystem

> Capstone tool ecosystem: integration de tools + MCP + A2A + routing + observability para production end-to-end. Components: (1) Tools (handlers), (2) Routing (cost/capability/LB), (3) Fallback chain (-single point of failure), (4) MCP (tools/resources/prompts), (5) A2A (multi-agent handoffs), (6) Observability (OTel metrics + tracing), (7) Audit log (+compliance). +Integrated, +Production, +Resilient, +Observable, +Reliable. Build process: (1) define tools, (2) register con cost + capability, (3) add routing, (4) add fallback, (5) add MCP, (6) add A2A, (7) add observability, (8) add audit, (9) test E2E (unit + integration), (10) production. SOTA 2024-25: MCP (standard tools +interop), A2A (multi-agent +standardized), LangGraph (state machines +cycles), Langfuse/LangSmith (observability +LLM-native), LiteLLM/Portkey (routing +fallback), OpenAI Agents SDK (handoffs +tools), Anthropic Skills (skills system +reusable), CrewAI (multi-agent +roles), OTLP (observability +standardized), OAuth 2.1 (auth +PKCE). Frameworks: mcp, a2a, langgraph, langfuse, langsmith, openai, anthropic, litellm, portkey, crewai. +Production: standard 2024-25. +Use cases: agent, RAG, automation, production. Trade-offs: cada uno + specialty, ecosystem + integrated, custom + simple. 2025: +MCP + A2A + native + ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/01-13/22
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ToolEcosystem class.
- Implementar register_tool, route, call, observe.
- Implementar fallback chain.
- Implementar audit log.
- Diagnosticar integration completa.

## Constrúyelo

```python
class ToolEcosystem:
    def route(self, capability):
        candidates = [t for t in self.tools.values() if t["capability"] == capability]
        return min(candidates, key=lambda t: t["cost"])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: capstone-tool
fase: 13
leccion: 23
---

1. Tools + routing.
2. Fallback.
3. MCP + A2A.
4. Observability.
5. +Production.
```

## Ejercicios

1. **Capstone**: integrar
   MCP + routing + observability.
2. **Multi-agent**: agregar
   A2A al ecosystem.
3. **Desafio**: full
   production deployment.

## Lecturas recomendadas

- "Building AI Applications" (LangChain, 2024)
- "Anthropic Skills + MCP" (Anthropic, 2024)
- "OpenAI Agents SDK" (OpenAI, 2024)
- "Langfuse + LangSmith" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Capstone Tool Ecosystem]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).