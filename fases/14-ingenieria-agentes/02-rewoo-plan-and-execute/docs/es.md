# ReWoo plan and execute

> ReWoo (Xu 2023, Reasoning Without Observation): plan-and-execute pattern. Planner genera DAG de tasks (#E1, #E2, #E3 con tool name + args, #E refs in args = dependencies) en un solo forward pass. Executor en topological order, resuelve #E refs con results, +parallel. +Token-efficient vs ReAct (no observation per step). Variants: ReWoo, ReAct, Plan-Execute, ADaPT. +Efficient, +Planning, +Parallel, +Token-efficient, +Reliable, +Scalable. Frameworks: langchain, langgraph, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, automation, multi-step. Decision: multi-step -> ReWoo, adaptive -> ReAct, simple -> Plan-Execute, token cost -> ReWoo, production -> ReWoo o ReAct. ReWoo vs ReAct: ReWoo (plan-and-execute -observation +token-efficient +parallel) vs ReAct (think-act-observe +observation +standard +adaptive). Trade-offs: ReWoo + efficient, ReAct + adaptive. 2025: +MCP + A2A + native + ReWoo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar parse_rewoo_plan (#E1, #E2).
- Implementar build_dag y topological_order.
- Implementar execute_plan con #E resolution.
- Diagnosticar ReWoo vs ReAct.

## Constrúyelo

```python
def parse_rewoo_plan(plan_text):
    pattern = r"#E(\d+)\s*=\s*(\w+)\[(.+?)\](?:\s*\(.*?\))?"
    for match in re.finditer(pattern, plan_text, re.DOTALL):
        task_id = int(match.group(1))
        tool = match.group(2)
        args = match.group(3)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: rewoo
fase: 14
leccion: 02
---

1. Plan-and-execute.
2. DAG #E1, #E2.
3. Topological order.
4. +Token-efficient.
5. +Parallel.
```

## Ejercicios

1. **ReWoo**: implementar
   ReWoo custom con
   HuggingFace.
2. **LangGraph**: probar
   LangGraph plan-execute.
3. **Desafio**: full
   multi-step agent.

## Lecturas recomendadas

- "ReWoo: Decoupling Reasoning from Observations for Efficient Augmented Language Models" (Xu et al., 2023)
- "Plan-and-Execute Agents" (LangChain Documentation)
- "LangGraph Plan-Execute" (LangChain, 2024)
- "ADaPT: As-needed Decomposition and Planning for Complex Tasks" (Prasad et al., 2023)

---

> 📚 **Adaptación al español de la lección [ReWoo Plan and Execute]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).