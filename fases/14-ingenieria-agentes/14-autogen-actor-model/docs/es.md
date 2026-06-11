# AutoGen actor model

> AutoGen (Microsoft 2024): actor model para multi-agent conversation. Components: (1) ConversableAgent (name + system_message + chat_history + send + receive + reply_count), (2) GroupChat (agents + max_round + messages + speaker_history, round-robin o manager), (3) UserProxyAgent (code execution, execute_code). +Conversational, +Multi-agent, +Code exec, +Manager, +Back-and-forth, +Termination, +Reliable, +Scalable, +Production. Variants: AutoGen seminal, ConversableAgent, GroupChat (round-robin, manager, selector), UserProxyAgent, custom. AutoGen vs CrewAI: AutoGen (+conversational +actor +code exec +Microsoft) vs CrewAI (+role-based +tasks +crews +sequential/parallel). Frameworks: autogen, microsoft, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, multi-agent, code, conversation, research. Decision: conversational -> AutoGen, role-based -> CrewAI, cycles -> LangGraph, handoffs -> OpenAI Agents, production -> combinacion. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ConversableAgent.
- Implementar UserProxyAgent con code exec.
- Implementar GroupChat con round-robin.
- Diagnosticar AutoGen vs CrewAI.
- Diagnosticar trade-offs.

## Constrúyelo

```python
class ConversableAgent:
    def receive(self, message, sender):
        self.chat_history.append({"role": "user", "content": message})
        reply = f"{self.name} replying to: {message[:50]}"
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: autogen
fase: 14
leccion: 14
---

1. ConversableAgent.
2. GroupChat.
3. UserProxyAgent.
4. Round-robin.
5. +Multi-agent.
```

## Ejercicios

1. **AutoGen**: usar AutoGen
   con group chat.
2. **Code exec**: implementar
   UserProxyAgent con code.
3. **Desafio**: multi-agent
   research con AutoGen.

## Lecturas recomendadas

- "AutoGen: Enabling Next-Gen LLM Applications" (Wu et al., Microsoft, 2024)
- "AutoGen Documentation" (Microsoft, 2024)
- "ConversableAgent Guide" (Microsoft, 2024)
- "Multi-Agent Conversation Patterns" (Microsoft, 2024)

---

> 📚 **Adaptación al español de la lección [AutoGen Actor Model]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).