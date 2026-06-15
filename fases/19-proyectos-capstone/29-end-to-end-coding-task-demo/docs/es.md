# 29 — End-to-end coding task demo

> End-to-end coding task: agent toma issue, navega codebase, edita, testea, abre PR. Eval sobre SWE-bench Verified (500 tasks reales). Métricas: pass rate, time, tokens, cost. Sistemas: Devin, Aider, OpenHands, Claude Code.

**Tipo:** Capstone
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 19/20-28
**Tiempo estimado:** 30 horas

## Objetivos

- Implementar agent end-to-end.
- Eval sobre SWE-bench Verified.
- Métricas (pass, time, cost).
- Comparar con baselines.

## El problema

End-to-end coding task: (1) Read issue. (2) Explore
codebase. (3) Plan changes. (4) Edit files. (5) Run
tests. (6) Iterate. (7) Open PR. Eval: SWE-bench
Verified (500 real GitHub issues con tests).
Sistemas: Devin (Cognition, 13.86% Verified),
Aider (95% Polyglot), OpenHands (18% Verified),
Claude Code (50%+ Verified, 2026). Métricas:
pass rate, time per task, tokens, cost.

## Constrúyelo

```python
class CodingAgent:
    def __init__(self, llm, tools, sandbox):
        self.llm = llm
        self.tools = tools
        self.sandbox = sandbox

    def solve(self, issue):
        # 1. Explore
        context = self.tools.explore(issue.repo)
        # 2. Plan
        plan = self.llm.plan(issue, context)
        # 3. Edit
        changes = self.tools.edit(plan)
        # 4. Test
        result = self.sandbox.test(changes)
        # 5. Iterate
        while not result["passes"] and result["iter"] < 5:
            changes = self.llm.fix(result, changes)
            result = self.sandbox.test(changes)
        return {"changes": changes, "result": result}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-coding-agent
fase: 19
leccion: 29
---

1. Issue understanding.
2. Codebase exploration.
3. Plan + edit.
4. Test + iterate.
5. SWE-bench eval.
```

## Ejercicios

1. **Mini-SWE**: 10 tasks
   simples.
2. **Eval**: pass rate
   métrica.
3. **Desafío**: 30% en
   SWE-bench Verified.

## Lecturas recomendadas

- "SWE-bench Verified"
  (Jimenez 2024, OpenAI 2024)
- "Aider Polyglot" (Gauthier 2025)
- "OpenHands" (Wang 2024)
- "Claude Code" (Anthropic 2025)

---

> 📚 **Adaptación al español** de la lección
> "[29-end-to-end-coding-task-demo]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
