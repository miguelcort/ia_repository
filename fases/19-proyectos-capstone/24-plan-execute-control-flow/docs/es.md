# 24 — Plan-execute control flow

> Plan-execute: el LLM genera un plan (lista de steps), executor corre cada step, evalúa resultado, replanifica si falla. Patrones: ReAct (reason+act), Plan-and-Execute, Tree of Thoughts. Frameworks: LangGraph, AutoGen.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 14, Fase 19/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Plan generator (LLM).
- Executor (tools).
- Replanner on failure.
- Eval (success rate, steps).

## El problema

Plan-execute vs ReAct: ReAct interleaves reasoning
y action (más flexible, menos predecible). Plan-
execute: (1) Plan: LLM genera lista de steps. (2)
Execute: corre cada step. (3) Observe: parse result.
(4) Replanify: si falla, regenerar plan. (5) Loop.
Más predecible, mejor para tasks estructurados.
LangGraph implementa plan-execute como graph state
machine. Frameworks: LangGraph, AutoGen, CrewAI.

## Constrúyelo

```python
class PlanExecuteAgent:
    def run(self, goal, max_replans=3):
        for _ in range(max_replans):
            plan = self.llm.plan(goal, self.context)
            results = []
            for step in plan:
                result = self.tools.execute(step)
                results.append(result)
                if self.is_failed(result):
                    break
            if self.all_succeeded(results):
                return self.synthesize(results)
            self.context.update(plan, results)
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
name: prompt-plan-execute
fase: 19
leccion: 24
---

1. Plan generator.
2. Step executor.
3. Replanner.
4. State machine.
5. Eval success rate.
```

## Ejercicios

1. **Plan-execute**: 5-step
   task.
2. **Replanner**: 3 replan
   max.
3. **Desafío**: LangGraph
   implementation.

## Lecturas recomendadas

- "Plan-and-Execute" (LangChain 2023)
- "ReAct" (Yao 2022)
- "LangGraph" (LangChain 2024)
- "Tree of Thoughts" (Yao 2023)

---

> 📚 **Adaptación al español** de la lección
> "[24-plan-execute-control-flow]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
