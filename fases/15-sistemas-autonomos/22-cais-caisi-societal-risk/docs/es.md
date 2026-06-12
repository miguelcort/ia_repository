# CAIS / CAISI societal risk

> CAIS + CAISI: (1) Center for AI Safety (non-profit+research), (2) Statements (extinction+responsible AI), (3) Risk frameworks (categories+severity), (4) Societal-scale (economy+geopolitics), (5) Extinction risk (catastrophic+all humanity), (6) Evaluation registry (independent+audit). 6 societal risk categories: extinction (catastrophic+all humanity), economic (high+workers+economies), geopolitical (high+nations+alliances), misinformation (high+democracy+public), bias (medium+minority+workers), concentration (medium+democracy+markets). assess_severity: order [medium, high, catastrophic] + max + iterate + highest. build_risk_register: per category Dict + stakeholders List + sort by severity desc [low, medium, high, catastrophic]. Criterios: CAIS = societal+statements+policy, METR = capability+time horizon+independent, RSP = ASL levels+compute+Anthropic, FSF = domains+critical caps+DeepMind. Decision: societal -> CAIS, capability -> METR, levels -> RSP, domains -> FSF, mix -> all. Frameworks: cais, metr, anthropic, deepmind, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/21
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SOCIETAL_RISK_CATEGORIES con 6 categories.
- Implementar CAIS_STATEMENTS con 2 statements.
- Implementar assess_severity con max.
- Implementar build_risk_register con sort.
- Diagnosticar CAIS vs METR vs RSP vs FSF.

## Constrúyelo

```python
SOCIETAL_RISK_CATEGORIES = {
    "extinction": {"name": "Extinction risk", "severity": "catastrophic", "stakeholders": ["all humanity"]},
    "economic": {"name": "Economic disruption", "severity": "high", "stakeholders": ["workers", "economies"]},
    # ... 6 categories
}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: cais-societal-risk
fase: 15
leccion: 22
---

1. 6 categories.
2. assess + register.
3. CAIS statements.
4. +Production.
```

## Ejercicios

1. **CAIS risks**: probar
   las 6 categories.
2. **assess_severity**: probar
   max.
3. **Desafio**: integrar
   con tu risk pipeline.

## Lecturas recomendadas

- "CAIS: Center for AI Safety" (CAIS, 2024)
- "AI Extinction Statement" (CAIS, 2023)
- "CAISI: Societal Risk" (CAISI, 2024)

---

> 📚 **Adaptación al español de la lección [CAIS / CAISI Societal Risk]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).