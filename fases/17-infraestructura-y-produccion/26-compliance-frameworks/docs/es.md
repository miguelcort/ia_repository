# Compliance frameworks

> Frameworks: (1) SOC2 (trust services), (2) HIPAA (healthcare), (3) GDPR (EU data), (4) EU AI Act (AI systems), (5) ISO 27001 (info security). FRAMEWORKS: SOC2 (name+description+scope+controls)+HIPAA+GDPR+EU_AI_ACT+ISO_27001. check_compliance: implemented (list)+score (implemented/total)+missing (required-implemented). audit_log_entry: action+user+resource+timestamp. Diferencias: SOC2 = trust+SaaS+US, HIPAA = healthcare+PHI+US, GDPR = EU+personal+privacy, EU AI Act = AI+risk-based+EU. Criterios: SOC2 = SaaS+trust+US, HIPAA = health+PHI+US, GDPR = EU+personal+privacy, EU AI Act = AI+risk+EU. Decision: SaaS -> SOC2, health -> HIPAA, EU -> GDPR, AI -> EU AI Act, mix -> all. Frameworks: openai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + compliance.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/25
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar FRAMEWORKS con 5 frameworks.
- Implementar check_compliance con score + missing.
- Implementar audit_log_entry.
- Diagnosticar frameworks.
- Diagnosticar criteria.

## Constrúyelo

```python
FRAMEWORKS = {
    "SOC2": {
        "name": "SOC 2",
        "scope": "Trust services criteria",
        "controls": ["security", "availability", "confidentiality"],
    },
    # ... 5 frameworks
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
name: compliance-frameworks
fase: 17
leccion: 26
---

1. 5 frameworks.
2. check_compliance.
3. audit_log_entry.
4. +Production.
```

## Ejercicios

1. **Frameworks**: probar
   los 5.
2. **Check**: probar
   score.
3. **Desafio**: integrar
   con SOC2 audit.

## Lecturas recomendadas

- "SOC 2" (AICPA, 2024)
- "GDPR" (EU, 2018)
- "EU AI Act" (EU, 2024)

---

> 📚 **Adaptación al español de la lección [Compliance Frameworks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).