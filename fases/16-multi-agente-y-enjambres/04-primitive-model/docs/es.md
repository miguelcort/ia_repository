# Primitive model

> Primitive agent: (1) Minimal (agent_id+name), (2) Handlers (on(performative, fn)), (3) State (set/get), (4) Tick (process inbox), (5) React (dispatch+send). PrimitiveAgent: agent_id (UUID)+name+inbox+outbox+state+handlers+on(perf, fn) register+send(receiver, perf, content) Message+tick() process inbox+receive(msg)+_dispatch(msg) lookup+call. Ventajas primitive: simple (few lines+clear), learning (understand+pedagogical), custom (tailored+no overhead), debug (trivial+no magic), lightweight (no deps+fast). Criterios: Primitive = learning+prototype+simple, CrewAI = roles+simple+crews, LangGraph = complex state+cycles+persistence, AutoGen = conversation+multi-actor+Microsoft. Decision: learn -> primitive, roles -> CrewAI, state -> LangGraph, convo -> AutoGen, mix -> LangGraph+custom. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + primitive.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PrimitiveAgent con agent_id + name + inbox + outbox.
- Implementar on(performative, fn) + handlers.
- Implementar send + receive + tick + _dispatch.
- Implementar state set/get.
- Diagnosticar primitive vs frameworks.

## Constrúyelo

```python
class PrimitiveAgent:
    def on(self, performative, fn):
        self.handlers[performative] = fn

    def send(self, receiver, performative, content):
        msg = Message(self.agent_id, receiver, content, performative=performative)
        self.outbox.append(msg)
        return msg

    def tick(self):
        while self.inbox:
            msg = self.inbox.pop(0)
            self._dispatch(msg)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: primitive-model
fase: 16
leccion: 04
---

1. PrimitiveAgent.
2. on + handlers.
3. tick + _dispatch.
4. +Production.
```

## Ejercicios

1. **PrimitiveAgent**: probar
   on + tick.
2. **Handlers**: probar
   dispatch.
3. **Desafio**: implementar
   2 agentes que se comunican.

## Lecturas recomendadas

- "Multi-Agent Systems" (Wooldridge, 2009)
- "JADE" (Bellifemine, 2007)
- "AgentSpeak" (Rao, 1996)

---

> 📚 **Adaptación al español de la lección [Primitive Model]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).