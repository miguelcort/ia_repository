# FinOps LLMs

> FinOps LLM: (1) Allocation (per team), (2) Budgets (cap), (3) Showback (visibility), (4) Chargeback (bill), (5) Unit econ (per token). CostRecord: team+service+cost_usd+tokens+period. FinOpsTracker: records+set_budget+record+total_cost (team, period)+cost_by_team (period)+budget_utilization (spent/budget)+unit_economics (cost/tokens). Diferencias: Showback = visibility+voluntary+awareness+internal, Chargeback = bill+mandatory+cost control+cross-team. Criterios: Showback = awareness+voluntary+internal, Chargeback = bill+mandatory+cross-team, FinOps = optimize+continuous+ROI. Decision: awareness -> showback, bill -> chargeback, optimize -> FinOps, mix -> all. Frameworks: custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + finops.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/26
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar CostRecord con team+service+cost+tokens+period.
- Implementar FinOpsTracker con records + budgets.
- Implementar total_cost + cost_by_team.
- Implementar budget_utilization + unit_economics.
- Diagnosticar showback vs chargeback.

## Constrúyelo

```python
def unit_economics(self, team, period):
    total_cost = self.total_cost(team=team, period=period)
    total_tokens = sum(r.tokens for r in self.records if r.team == team and r.period == period)
    if total_tokens == 0:
        return 0.0
    return total_cost / total_tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: finops-llms
fase: 17
leccion: 27
---

1. CostRecord.
2. FinOpsTracker.
3. budget_utilization.
4. unit_economics.
5. +Production.
```

## Ejercicios

1. **Tracker**: probar
   record + total.
2. **Unit**: probar
   per-token cost.
3. **Desafio**: integrar
   con OpenAI billing.

## Lecturas recomendadas

- "FinOps Framework" (FinOps, 2024)
- "Cloud FinOps" (Economics, 2024)
- "LLM Cost Optimization" (Anyscale, 2024)

---

> 📚 **Adaptación al español de la lección [FinOps LLMs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).