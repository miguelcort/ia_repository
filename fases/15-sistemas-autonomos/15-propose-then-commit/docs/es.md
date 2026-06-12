# Propose-then-commit

> Propose-then-commit: (1) Plan generation (steps+describe), (2) User approval (manual+auto), (3) Commit (execute+rollback). Used in coding agents: Aider, Claude Code, Cline, Cursor. ProposeThenCommit: pending dict (ID+action), committed dict (approver+time), rejected dict (reason+time), propose(action, payload, summary) UUID+pending, approve(proposal_id, approver) pop pending+store committed, reject(proposal_id, reason) pop pending+store rejected. Plan: steps list (dict+ID), add_step(desc, action, args, requires_approval) description+action+args+bool, describe() format+[A] marker, needs_approval_count sum, auto_steps filter+not requires, approval_steps filter+requires. commit_with_rollback: try plan.steps -> if requires_approval propose+approve+executor -> except rollback. Criterios: propose-then-commit = dangerous+audit+review, direct = safe+fast+read-only. Decision: DB writes -> propose, file ops -> propose, read-only -> direct, production -> mix. Frameworks: aider, anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar ProposeThenCommit con pending + committed + rejected.
- Implementar propose + approve + reject.
- Implementar Plan con steps + add_step + describe.
- Implementar commit_with_rollback.
- Diagnosticar propose vs direct.

## Constrúyelo

```python
class ProposeThenCommit:
    def approve(self, proposal_id, approver="user"):
        if proposal_id not in self.pending:
            raise KeyError(f"unknown proposal: {proposal_id}")
        proposal = self.pending.pop(proposal_id)
        self.committed[proposal_id] = {**proposal, "approver": approver, "committed_at": time.time()}
        return proposal
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: propose-then-commit
fase: 15
leccion: 15
---

1. ProposeThenCommit.
2. Plan + steps.
3. Commit + rollback.
4. +Production.
```

## Ejercicios

1. **ProposeThenCommit**: probar
   approve + reject.
2. **Plan**: probar
   requires_approval.
3. **Desafio**: integrar
   con Aider o Claude Code.

## Lecturas recomendadas

- "Aider: Architect Mode" (Aider, 2024)
- "Claude Code: Plan Mode" (Anthropic, 2024)
- "Propose-Verify Pattern" (Suno, 2024)

---

> 📚 **Adaptación al español de la lección [Propose-then-commit]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).