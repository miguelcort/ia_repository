# Society of mind debate

> SoM debate: (1) Minsky (heterogeneous+society), (2) Debate (multi-round+refine), (3) Perspectives (diverse+optimist+pess+neutral), (4) Voting (majority+or synthesis). DebateAgent: agent_id+perspective+stance_fn+argue(topic). debate: agents list+rounds int+prior history+iterate (for round, for agent, stance). majority_vote: counts Dict+max key. synthesize: unique Set+joined [perspective] stance. Ventajas debate vs single-shot: refinement (multi-round+update), diversity (multiple perspectives+diverse), robust (outliers+voting), transparency (show reasoning+audit), discovery (new options+synthesis). Criterios: Debate = refine+audit+multi-round, Ensemble = many models+voting+bagging, Single-shot = fast+simple+cheap, SoM = heterogeneous+diverse+Minsky. Decision: refine -> debate, many -> ensemble, fast -> single, diverse -> SoM. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + debate.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar DebateAgent con perspective + stance_fn.
- Implementar debate con rounds + prior.
- Implementar majority_vote + synthesize.
- Diagnosticar refinement.
- Diagnosticar debate vs other.

## Constrúyelo

```python
def debate(topic, agents, rounds=2):
    statements = []
    history = {a.agent_id: [] for a in agents}
    for r in range(rounds):
        round_statements = []
        for a in agents:
            prior = [s for h in history.values() for s in h]
            stance = a.stance_fn(topic, prior=prior)
            stmt = {"agent_id": a.agent_id, "stance": stance, "round": r}
            round_statements.append(stmt)
            history[a.agent_id].append(stance)
        statements.extend(round_statements)
    return statements
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: society-of-mind-debate
fase: 16
leccion: 07
---

1. DebateAgent.
2. debate + rounds.
3. majority + synthesize.
4. +Production.
```

## Ejercicios

1. **DebateAgent**: probar
   argue + stance.
2. **debate**: probar
   multi-round.
3. **Desafio**: integrar
   con CrewAI debate.

## Lecturas recomendadas

- "Society of Mind" (Minsky, 1986)
- "Multi-Agent Debate" (Du et al., 2023)
- "Self-Consistency" (Wang et al., 2022)

---

> 📚 **Adaptación al español de la lección [Society of Mind Debate]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).