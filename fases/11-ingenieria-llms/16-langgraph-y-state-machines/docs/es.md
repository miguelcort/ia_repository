# LangGraph y state machines

> LangGraph (LangChain 2024): framework para stateful, multi-actor workflows. StateGraph: nodes (funciones que procesan state y retornan update), edges (connections entre nodes, conditional para branches), state (shared dict con channels, reducers). Memory: InMemorySaver (dev), SqliteSaver (local), PostgresSaver (production), RedisSaver (distributed). Threads: thread_id per user, multi-tenant. Use cases: chatbots, RAG agents, multi-actor, human-in-loop, +memory, +reasoning. Vs LangChain Agents: LangGraph es lower-level, +flexibility, stateful. Frameworks: LangGraph, LangGraph Cloud (managed), LangGraph Studio (visual IDE). Memory persistente: checkpointers, restore, time-travel debugging. SOTA 2024-25: LangGraph + LangSmith + memory + observability + reasoning (o1-style) + multimodal + MCP + tools.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/09-function-calling, 11/13-aplicacion-de-produccion
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar node y edge.
- Implementar state init y step.
- Build simple graph.
- Diagnosticar checkpointers.
- Diagnosticar memory persistente.

## Constrúyelo

```python
def build_simple_graph():
    nodes = {
        "input": lambda s: {**s, "step": 1},
        "process": lambda s: {**s, "step": 2, "result": "done"},
        "output": lambda s: {**s, "done": True, "step": 3},
    }
    edges = [("input", "process", None), ("process", "output", None)]
    return nodes, edges
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: langgraph
fase: 11
leccion: 16
---

1. StateGraph nodes + edges.
2. State, channels, reducers.
3. Memory: PostgresSaver.
4. Threads multi-tenant.
5. LangGraph Studio.
```

## Ejercicios

1. **Graph**: implementar agent
   graph con LangGraph.
2. **Memory**: agregar
   PostgresSaver.
3. **Desafio**: multi-actor
   + human-in-loop.

## Lecturas recomendadas

- "LangGraph: Stateful Multi-Actor Applications" (LangChain, 2024)
- "Building Agentic Apps with LangGraph" (LangChain, 2024)
- "LangGraph Studio" (LangChain, 2024)
- "Memory in LangGraph" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [LangGraph State Machines]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).