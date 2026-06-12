# Voting debate topology

> Voting + debate: (1) Weighted (per-agent+sum), (2) Debate (rounds+refine), (3) Role weight (per-role+custom), (4) Abstain (skip+neutral), (5) Topology (ring/star/FC). weighted_vote: scores Dict+max. approve_vote: count yes Sum+> threshold fraction. ring_topology: 2 neighbors (i, i+1)+mod n. star_topology: hub 0. fully_connected: all pairs (for i+for j). debate_topology: per round neighbors combined+history. Ventajas topologies: Ring = scale+sequential+cheap, Star = central+fast+hub, FC = robust+no bottleneck+costly. Criterios: Weighted = roles+expertise+custom, Approve = simple+yes/no+threshold, Topology = structure+pattern+communication. Decision: roles -> weighted, simple -> approve, structure -> topology, mix -> weighted+topology. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + voting.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar weighted_vote + approve_vote.
- Implementar ring_topology + star_topology + fully_connected_topology.
- Implementar debate_topology con rounds.
- Diagnosticar topologies.
- Diagnosticar voting criteria.

## Constrúyelo

```python
def ring_topology(n):
    return [(i, (i + 1) % n) for i in range(n)]

def star_topology(n):
    hub = 0
    return [(hub, i) for i in range(1, n)]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: voting-debate-topology
fase: 16
leccion: 15
---

1. weighted + approve.
2. ring + star + FC.
3. debate_topology.
4. +Production.
```

## Ejercicios

1. **weighted_vote**: probar
   per-agent.
2. **Topology**: probar
   ring + star.
3. **Desafio**: implementar
   abstention.

## Lecturas recomendadas

- "Multi-Agent Voting" (Conitzer, 2012)
- "Topology Patterns" (Hohpe, 2003)
- "Debate Networks" (Du et al., 2023)

---

> 📚 **Adaptación al español de la lección [Voting Debate Topology]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).