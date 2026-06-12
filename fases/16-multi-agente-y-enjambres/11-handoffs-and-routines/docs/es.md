# Handoffs and routines

> Handoffs + routines: (1) Transfer (control+from-to), (2) Multi-step (sequence+steps), (3) OpenAI Agents SDK (pattern+standard), (4) Routing (logic+reason), (5) Shared context (state+context). Handoff: from_agent+to_agent+reason+context+id (UUID). Routine: name+steps list+cursor+current_step() None si done+advance(result) history+increment+is_done() cursor>=len+run(executor) for step+executor+history. Ventajas routines: predictable (known sequence+same each time), reusable (template+once-defined), auditable (steps+history), composable (combine+module), comprehension (readable+clear). Criterios: Handoffs = explicit+agent-initiated+reason, Routing = by condition+automatic+conditional, Supervisor = central+simple+single. Decision: explicit -> handoffs, condition -> routing, central -> supervisor, mix -> handoffs+supervisor. Frameworks: openai, langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + handoffs.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/10
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Handoff con from_agent + to_agent + reason + context.
- Implementar Routine con steps + cursor + advance.
- Implementar current_step + is_done + run.
- Diagnosticar routines.
- Diagnosticar handoffs vs routing.

## Constrúyelo

```python
class Routine:
    def __init__(self, name, steps):
        self.name = name
        self.steps = list(steps)
        self.cursor = 0
        self.history = []

    def run(self, executor):
        while not self.is_done():
            step = self.current_step()
            result = executor(step)
            self.advance(result)
        return self.history
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: handoffs-and-routines
fase: 16
leccion: 11
---

1. Handoff + Routine.
2. cursor + advance.
3. run(executor).
4. +Production.
```

## Ejercicios

1. **Handoff**: probar
   from + to.
2. **Routine**: probar
   run.
3. **Desafio**: integrar
   con OpenAI Agents SDK.

## Lecturas recomendadas

- "OpenAI Agents SDK" (OpenAI, 2024)
- "Handoffs Pattern" (OpenAI, 2024)
- "Anthropic: Multi-Agent" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Handoffs and Routines]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).