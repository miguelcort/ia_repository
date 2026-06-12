# Supervisor orchestrator pattern

> Supervisor: (1) Central (coordinator+single point), (2) Dispatch (workers+roles), (3) Collect (results+aggregate), (4) Plan + monitor (steps+state), (5) LangGraph + A2A (state+protocol). Supervisor: workers dict+register_worker(worker, role) role key+dispatch(task, role) lookup+execute+_select_role(task) action matching (code->developer, test->tester, review->reviewer)+run_plan(plan) iterate+dispatch. Worker: agent_id (UUID)+role+assign(task) execute+_execute done:action. Roles tipicos: Developer (code+implement+build), Tester (test+QA+validate), Reviewer (review+audit+quality), Worker (default+generic). Criterios: Supervisor = central+simple+single point, Hierarchical = multi-level+tree+delegation, Society-of-mind = debate+diverse+Minsky, A2A = peer-to-peer+no central+modern. Decision: simple -> supervisor, multi-level -> hierarchical, diverse -> SoM, peer -> A2A. Frameworks: langgraph, anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + supervisor.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Worker con agent_id + role + assign.
- Implementar Supervisor con workers dict + register_worker.
- Implementar dispatch + _select_role + run_plan.
- Diagnosticar roles.
- Diagnosticar supervisor vs other.

## Constrúyelo

```python
class Supervisor:
    def dispatch(self, task, role=None):
        if role is None:
            role = self._select_role(task)
        worker = self.workers.get(role)
        if not worker:
            raise ValueError(f"no worker for role: {role}")
        result = worker.assign(task)
        self.completed.append({"task": task, "role": role, "result": result})
        return result
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: supervisor-orchestrator
fase: 16
leccion: 05
---

1. Supervisor + Worker.
2. dispatch + _select_role.
3. run_plan.
4. +Production.
```

## Ejercicios

1. **Worker**: probar
   assign + execute.
2. **Supervisor**: probar
   dispatch + run_plan.
3. **Desafio**: integrar
   con LangGraph.

## Lecturas recomendadas

- "LangGraph: Supervisor" (LangChain, 2024)
- "Multi-Agent Supervisor" (Anthropic, 2024)
- "A2A Protocol" (Google, 2025)

---

> 📚 **Adaptación al español de la lección [Supervisor Orchestrator Pattern]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).