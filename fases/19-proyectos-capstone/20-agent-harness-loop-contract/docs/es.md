# 20 — Agent harness loop contract

> Agent harness: el loop plan-act-observe-recover con presupuesto acotado. Componentes: state (TodoWrite), tool dispatcher, observation budget, recovery. Estándar 2026: Claude Code, OpenCode, Cursor Composer 2.

**Tipo:** Construir
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 14, Fase 13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar loop plan-act-observe-recover.
- State management (TodoWrite).
- Budget tracking (turns, tokens, dollars).
- Recovery patterns.

## El problema

El agent harness 2026 tiene cuatro componentes:
**Plan**: estado (TodoWrite) que el modelo reescribe
cada turno. **Act**: dispatcher de tools. **Observe**:
parse output, truncate. **Recover**: handle errors,
rollback, retry. Contrato: el harness expone tools,
el modelo decide tool calls, el harness ejecuta
y devuelve result. Budget: max_turns, max_tokens,
max_dollars. Sobre budget → stop.

## Constrúyelo

```python
class AgentHarness:
    def __init__(self, llm, tools, budget):
        self.llm = llm
        self.tools = tools
        self.budget = budget
        self.state = {"todos": [], "history": []}

    def run(self, prompt, max_turns=20):
        for turn in range(max_turns):
            if self.budget.exceeded():
                break
            # Plan
            self.state = self.llm.update_plan(prompt, self.state)
            # Act
            tool_call = self.llm.decide_tool(self.state)
            if tool_call is None:
                return self.state["final_answer"]
            # Observe
            result = self.tools.execute(tool_call)
            self.state["history"].append((tool_call, result))
            # Recover
            if "error" in result:
                self.state = self.recover(tool_call, result)
        return self.state.get("final_answer", "")
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-agent-harness
fase: 19
leccion: 20
---

1. Plan: state TodoWrite.
2. Act: tool dispatcher.
3. Observe: truncate output.
4. Recover: error handling.
5. Budget: turns, tokens, dollars.
```

## Ejercicios

1. **Loop**: implementar harness
   con 5 tools.
2. **Budget**: enforce max_turns.
3. **Desafío**: recovery
   pattern para timeout.

## Lecturas recomendadas

- "Claude Code" (Anthropic 2024)
- "OpenCode" (2024)
- "Cursor Composer 2" (Cursor 2026)

---

> 📚 **Adaptación al español** de la lección
> "[20-agent-harness-loop-contract]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
