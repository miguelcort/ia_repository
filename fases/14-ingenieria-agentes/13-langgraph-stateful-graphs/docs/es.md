# LangGraph stateful graphs

> LangGraph (LangChain 2024): stateful graphs para agents. Components: (1) State (dict + snapshot + history), (2) Node (name + fn), (3) Edge (src + dst + condition function(state) -> bool), (4) Graph (entry + run from entry + max_steps). Conditional edges: function(state) -> bool, True -> take edge, False -> try next. +Routing, +Branching, +Decisions. +Stateful, +Cycles, +Conditional, +Persistent, +Reliable, +Flexible, +Production. Variants: LangGraph, DAG, Airflow, Prefect, Dagster, custom SM, Temporal. LangGraph vs DAG: LangGraph (+cycles +stateful +conditional +persistent) vs DAG (+acyclic +simple +parallel -cycles). Frameworks: langgraph, airflow, dagster, prefect, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, multi-agent, ETL, cycles, branching, state. Decision: cycles -> LangGraph, acyclic -> DAG, custom -> custom SM, production -> LangGraph. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + LangGraph.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar State class.
- Implementar Node y Edge.
- Implementar Graph con conditional edges.
- Diagnosticar LangGraph vs DAG.
- Diagnosticar conditional routing.

## Constrúyelo

```python
class Graph:
    def step(self, current, state):
        result = self.nodes[current].run(state)
        for edge in self.edges[current]:
            if edge.condition is None or edge.condition(state):
                return edge.dst
        return None
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
fase: 14
leccion: 13
---

1. Nodes + edges.
2. State + cycles.
3. Conditional edges.
4. Persistence.
5. +Production.
```

## Ejercicios

1. **LangGraph**: usar
   LangGraph con cycles.
2. **Conditional**: implementar
   branching custom.
3. **Desafio**: full
   multi-agent LangGraph.

## Lecturas recomendadas

- "LangGraph Documentation" (LangChain, 2024)
- "LangGraph: Stateful Graphs for Agents" (LangChain, 2024)
- "Temporal: Workflow as Code" (Temporal, 2024)
- "Building Stateful Agents" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [LangGraph Stateful Graphs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).