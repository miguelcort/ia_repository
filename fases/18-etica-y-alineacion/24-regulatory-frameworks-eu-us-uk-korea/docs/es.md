# 24 — Regulatory frameworks: EU AI Act, US EO, UK, Korea

> Marcos regulatorios: EU AI Act (2024, risk-based), US Executive Order 14110 (2023), UK AI Safety Institute (2023), Korea AI Act (2025). Compliance: documentación, evals, transparency, human oversight.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/18, 18/26
**Tiempo estimado:** ~30 minutos

## Objetivos

- Conocer los 4 marcos regulatorios principales.
- Mapear sistema a nivel de riesgo.
- Implementar documentation templates.
- Diagnosticar compliance gaps.

## Constrúyelo

```python
def eu_ai_act_risk_level(system):
    """EU AI Act: unacceptable, high, limited, minimal."""
    if system.use_case in ["social_scoring", "biometric_tracking"]:
        return "unacceptable"
    if system.use_case in ["hiring", "credit", "medical", "education"]:
        return "high"
    if system.use_case in ["chatbot", "deepfake"]:
        return "limited"
    return "minimal"


def compliance_checklist(system, framework):
    """Checklist por framework."""
    if framework == "EU_AI_Act":
        return {
            "risk_classification": eu_ai_act_risk_level(system),
            "data_governance": system.data_documented,
            "transparency": system.disclosure_mechanism,
            "human_oversight": system.human_in_loop,
            "accuracy_robustness": system.evals_passed,
            "technical_documentation": system.tech_doc_complete,
        }
    if framework == "US_EO_14110":
        return {
            "red_team_results": system.redteam_done,
            "capability_reporting": system.capability_disclosed,
        }


def audit_trail(action, model_id, timestamp, user_id):
    """EU AI Act requirement: log all decisions."""
    return {
        "action": action,
        "model": model_id,
        "time": timestamp,
        "user": user_id,
        "compliance_id": f"AUDIT-{timestamp}",
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
name: prompt-regulatory
fase: 18
leccion: 24
---

1. Mapear sistema a risk level.
2. Documentation per framework.
3. Audit trail en producción.
4. Compliance reporting.
```

## Ejercicios

1. **Risk classification**: clasificar
   tu sistema bajo EU AI Act.
2. **Documentation**: completar tech
   doc template.
3. **Desafío**: diseñar compliance
   pipeline para organización.

## Lecturas recomendadas

- "EU AI Act" (Parlamento Europeo 2024)
- "US Executive Order 14110" (2023)
- "UK AI Safety Institute" (2023)
- "Korea AI Basic Act" (2025)

---

> 📚 **Adaptación al español** de la lección
> "[24-regulatory-frameworks-eu-us-uk-korea]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
