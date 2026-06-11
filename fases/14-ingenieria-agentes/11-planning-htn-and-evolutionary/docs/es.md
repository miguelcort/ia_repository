# Planning HTN and evolutionary

> HTN (Hierarchical Task Network): hierarchical task decomposition. Methods (task -> subtasks) + recursive decomposition. Primitive = leaf executable. Plan = sequence of primitives. +Structured, +Hierarchical, +Modular, +Composable, +Reliable. Evolutionary planning: population (random plans) + evaluate (fitness function score) + select (top, elitism) + mutate (swap, change, insert) + repeat N generations. +Optimization, +Convergence, +Quality, +Reliable. Variants: HTN, evolutionary (GA, ES, NEAT), ADaPT (Prasad 2023 +adaptive +decomposition +LLM), RAP (Hao 2023 +reasoning +action +plan), SayCan (Ahn 2022 +grounding +affordance +robotics), SayPlan, LLM planner. Frameworks: pyhop, planner, langchain, langgraph, smolagents, DEAP. +Production: standard 2024-25. +Use cases: agent, RAG, automation, planning, robotics, code. Decision: structured -> HTN, optimization -> evolutionary, adaptive -> ADaPT, reasoning -> RAP, robotics -> SayCan, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + planning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar htn_decompose y htn_plan.
- Implementar evolutionary_plan con population + mutation.
- Implementar llm_plan.
- Diagnosticar HTN vs evolutionary vs ADaPT vs RAP.

## Constrúyelo

```python
def htn_plan(task, methods, max_depth=10):
    if max_depth <= 0:
        return [task]
    subtasks = htn_decompose(task, methods)
    plan = []
    for sub in subtasks:
        if sub in methods:
            plan.extend(htn_plan(sub, methods, max_depth - 1))
        else:
            plan.append(sub)
    return plan
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: planning
fase: 14
leccion: 11
---

1. HTN decompose.
2. Evolutionary mutate.
3. LLM planning.
4. +Hierarchical.
5. +Optimization.
```

## Ejercicios

1. **HTN**: implementar
   HTN custom con pyhop.
2. **Evolutionary**: probar
   evolutionary planning.
3. **Desafio**: LLM
   planning con RAP.

## Lecturas recomendadas

- "Hierarchical Task Network Planning" (Erol et al., 1994)
- "SayCan: Do As I Can, Not As I Say" (Ahn et al., 2022)
- "Reasoning with Language Model is Planning with World Model" (Hao et al., 2023)
- "ADaPT: As-needed Decomposition and Planning" (Prasad et al., 2023)

---

> 📚 **Adaptación al español de la lección [Planning HTN and Evolutionary]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).