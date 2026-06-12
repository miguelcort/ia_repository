# SRE for AI

> SRE AI: (1) SLIs (indicators), (2) SLOs (objectives), (3) Error budgets (allowed), (4) Incident (response), (5) On-call (rotation), (6) Postmortem (blameless). SLI: name+good_events+total_events+value() (good/total). SLO: name+sli+target+window_days+met() (sli >= target)+error_budget() (max(0, 1-target)). burn_rate: recent_value+window_days. Incident: title+severity+started_at+resolve+attach_postmortem+duration_minutes. severity_rank. Principles SRE AI: SLI/SLO (indicators+objectives), Error budget (allowed fail), Blameless (no blame), Automation (reduce toil), On-call (rotation). Criterios: SRE = production+critical+reliable, Ad-hoc = dev+cheap+small. Decision: prod -> SRE, dev -> ad-hoc, mix -> both. Frameworks: prometheus, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + sre.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/22
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SLI con good + total.
- Implementar SLO con target + met + error_budget.
- Implementar burn_rate.
- Implementar Incident con resolve + postmortem.
- Diagnosticar SRE principles.

## Constrúyelo

```python
class SLO:
    def error_budget(self):
        return max(0, 1 - self.target)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: sre-for-ai
fase: 17
leccion: 23
---

1. SLI + SLO.
2. error_budget.
3. burn_rate.
4. Incident.
5. +Production.
```

## Ejercicios

1. **SLI/SLO**: probar
   met + budget.
2. **Burn**: probar
   rate.
3. **Desafio**: integrar
   con PagerDuty.

## Lecturas recomendadas

- "SRE Book" (Google, 2017)
- "Error Budgets" (Google SRE, 2017)
- "Blameless Postmortem" (Google SRE, 2017)

---

> 📚 **Adaptación al español de la lección [SRE for AI]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).