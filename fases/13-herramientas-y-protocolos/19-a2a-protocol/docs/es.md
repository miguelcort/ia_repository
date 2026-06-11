# A2A protocol

> A2A (Google 2025, Agent-to-Agent): protocol para comunicacion agent-to-agent. Agent cards: discovery metadata (name, version, description, capabilities: streaming/tools/push notifications, skills list, url endpoint). JSON-RPC 2.0 + streaming (SSE). Multi-agent orchestration. Methods: (1) tasks/send (send task, return task_id), (2) tasks/get (status, result, error), (3) tasks/cancel (abort, -if completed), (4) streaming/subscribe (SSE updates, +real-time), (5) push_notification (webhook, +async). +SOTA 2024-25. +Standardized, +Interop, +Composable, +Streaming, +Discoverable. Frameworks: a2a, google, anthropic, openai, langchain, MCP, ANP. +Production: standard 2024-25. +Use cases: agent, RAG, automation, multi-agent, ecosystem. Decision: agent-to-agent -> A2A, tools -> MCP, custom -> custom protocol, production -> A2A + MCP. Variants: A2A, MCP, custom, ANP, LangChain. Trade-offs: A2A + interop, MCP + tools, custom + simple. 2025: +A2A + MCP + native + multi-agent + ecosystem.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/06, 13/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_agent_card.
- Implementar make_a2a_message.
- Implementar a2a_task_send, a2a_task_get, a2a_task_cancel.
- Implementar a2a_stream_subscribe.
- Diagnosticar A2A vs MCP vs custom.

## Constrúyelo

```python
def make_agent_card(name, version, description, capabilities, skills, url):
    return {
        "name": name,
        "version": version,
        "description": description,
        "capabilities": capabilities,
        "skills": skills,
        "url": url,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: a2a
fase: 13
leccion: 19
---

1. Agent-to-agent.
2. Agent cards.
3. JSON-RPC + streaming.
4. tasks/* + streaming.
5. +SOTA 2024-25.
```

## Ejercicios

1. **A2A**: implementar
   A2A client con a2a SDK.
2. **Streaming**: probar
   streaming/subscribe.
3. **Desafio**: multi-agent
   orchestration.

## Lecturas recomendadas

- "A2A Protocol Specification" (Google, 2025)
- "A2A Python SDK" (https://github.com/google/a2a)
- "Multi-Agent Systems" (Wooldridge, 2009)
- "Agent Communication" (FIPA ACL, 2002)

---

> 📚 **Adaptación al español de la lección [A2A Protocol]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).