# Agent workbench capstone

> Capstone: (1) Full integration (all parts+compose), (2) Tools + memory + plan (registry+plan), (3) Reviewer + gates (post-hoc+pre-deploy), (4) Handoff + scope + todos (multi-session+scope+inspection), (5) Stats (languages+files), (6) End-to-end (full workflow+production). CapstoneWorkbench: registry+memory+scope+reviewer+gate+inspect_repo(root) todos+stats+execute_action(action) scope check+review_output(output) reviewer+run_gates(fn) gate+full_workflow(root, action, output) inspect+execute+review+done. Flujo: (1) inspect_repo -> todos+stats, (2) execute_action -> scope validate+history, (3) review_output -> reviewer+approved, (4) full_workflow -> inspect+execute+review+done. Criterios: Capstone = complex+integrated+multi-step+production, Individual = simple+focused+single concern+learning. Decision: complex -> capstone, simple -> individual, mix -> both, production -> capstone. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + workbench.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/41
**Tiempo estimado:** ~60 minutos

## Objetivos

- Implementar CapstoneWorkbench que compone registry+memory+scope+reviewer+gate.
- Implementar inspect_repo + execute_action + review_output + run_gates.
- Implementar full_workflow end-to-end.
- Diagnosticar integracion.
- Diagnosticar capstone vs individual.

## Constrúyelo

```python
class CapstoneWorkbench:
    def full_workflow(self, root, action, output):
        inspect = self.inspect_repo(root)
        execution = self.execute_action(action)
        if not execution["ok"]:
            return {"step": "execute", "ok": False, "reason": execution["reason"]}
        review = self.review_output(output)
        return {
            "step": "done",
            "ok": review["approved"],
            "inspect": inspect,
            "execution": execution,
            "review": review,
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
name: workbench-capstone
fase: 14
leccion: 42
---

1. CapstoneWorkbench.
2. inspect + execute + review.
3. full_workflow.
4. +Production.
```

## Ejercicios

1. **Capstone**: probar
   todos los modulos.
2. **Workflow**: probar
   end-to-end.
3. **Desafio**: agregar
   handoff + checkpoint.

## Lecturas recomendadas

- "Anthropic: Building Effective Agents" (Anthropic, 2024)
- "LangGraph: StateGraph" (LangChain, 2024)
- "Production Agents" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Agent Workbench Capstone]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).