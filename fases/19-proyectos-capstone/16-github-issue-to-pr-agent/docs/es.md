# 16 — GitHub issue to PR agent

> GitHub issue-to-PR agent (Sweep, Devin, Codex): tomar issue, plan, branch, código, tests, abrir PR. Sistemas complejos: requiere understanding del codebase, navigation, y safe operations.

**Tipo:** Capstone
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 11, Fase 14 (agents), Fase 13
**Tiempo estimado:** 25 horas

## Objetivos

- Parse GitHub issue.
- Plan, branch, code, test.
- Open PR con descripción.
- Eval sobre SWE-bench.

## El problema

GitHub issue-to-PR agents automatizan el ciclo
issue → PR: (1) Parse issue (title, body, labels).
(2) Plan (files to modify). (3) Branch + checkout.
(4) Code changes (LLM + tools). (5) Test generation.
(6) Run tests. (7) Commit. (8) Open PR con
descripción generada. (9) Iterate sobre CI failures.
Sistemas: Sweep, Devin, Codex, Aider. Eval: SWE-
bench (2K+ tasks reales). Métricas: % issues
resueltos, % PRs merged, time per issue.

## Constrúyelo

```python
class IssueToPRAgent:
    def run(self, issue_id):
        issue = github.get_issue(issue_id)
        plan = self.llm.plan(issue, repo_context)
        branch = self.create_branch(plan.target_files)
        changes = self.apply_changes(plan)
        tests = self.generate_tests(changes)
        self.run_tests(tests)
        return self.open_pr(branch, changes, plan)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-issue-to-pr
fase: 19
leccion: 16
---

1. Parse issue + repo context.
2. Plan archivos a modificar.
3. Branch + code + tests.
4. Run CI.
5. Open PR.
```

## Ejercicios

1. **Issue parser**: 100 issues
   de open source.
2. **Agent loop**: 10 issues
   end-to-end.
3. **Desafío**: SWE-bench
   pass rate > 30%.

## Lecturas recomendadas

- "SWE-bench" (Jimenez 2024)
- "Sweep" (2023)
- "Devin" (Cognition 2024)
- "Aider" (Gauthier 2024)

---

> 📚 **Adaptación al español** de la lección
> "[16-github-issue-to-pr-agent]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
