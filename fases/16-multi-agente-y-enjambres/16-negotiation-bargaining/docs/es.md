# Negotiation bargaining

> Negotiation: (1) Utility (range+per-agent), (2) TIOLI (accept or leave+one-shot), (3) Alternating (multi-round+converge), (4) ZOPA (zone+possible agreement), (5) Nash (optimal+maximize product). zopa_check: max(low)+min(high)+if low<=high overlap. nash_bargaining: utilities (a_disagreement+b_disagreement)+max product a*b. alternating_offers: start from a+rounds+alternate (r%2)+in range both. Ventajas Nash vs TIOLI: optimal (max product+Pareto), fair (symmetric+disagreement), Pareto (efficient+no waste), information (ranges+disagreement), reciprocity (mutual benefit+balance). Criterios: TIOLI = power+fast+one-shot, Alternating = cooperative+multi-round+converge, Nash = optimal+max product+symmetric, Mediator = complex+multi-party+external. Decision: power -> TIOLI, coop -> alternating, optimal -> Nash, complex -> mediator. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + negotiation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/15
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar zopa_check con overlap.
- Implementar nash_bargaining con max product.
- Implementar alternating_offers con rounds.
- Implementar take_it_or_leave_it.
- Diagnosticar TIOLI vs Nash.
- Diagnosticar criteria.

## Constrúyelo

```python
def zopa_check(agent_a_range, agent_b_range):
    low = max(agent_a_range[0], agent_b_range[0])
    high = min(agent_a_range[1], agent_b_range[1])
    if low <= high:
        return True, (low, high)
    return False, None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: negotiation-bargaining
fase: 16
leccion: 16
---

1. zopa + nash.
2. alternating + tioli.
3. utilities.
4. +Production.
```

## Ejercicios

1. **zopa**: probar
   overlap.
2. **nash**: probar
   max product.
3. **Desafio**: implementar
   mediator.

## Lecturas recomendadas

- "Nash Bargaining" (Nash, 1950)
- "Multi-Agent Negotiation" (Jennings, 2001)
- "Contract Net" (Smith, 1980)

---

> 📚 **Adaptación al español de la lección [Negotiation Bargaining]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).