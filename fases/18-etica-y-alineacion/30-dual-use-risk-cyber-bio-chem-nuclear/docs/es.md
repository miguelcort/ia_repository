# 30 — Dual-use risk: cyber, bio, chem, nuclear

> Riesgo de uso dual: modelos que pueden asistir ataques cyber (CVE generation, exploit writing), bio (pathogen synthesis, pandemic potential), chem (drug synthesis, weaponization), nuclear. Requiere threat modeling, red-teaming, controls.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/17, 18/18
**Tiempo estimado:** ~30 minutos

## Objetivos

- Mapear dominios de uso dual.
- Implementar threat model.
- Evaluar uplift (cuánto el modelo aumenta capacidad del atacante).
- Diagnosticar controls.

## Constrúyelo

```python
def dual_use_risk_score(model, domain):
    """Risk score por dominio."""
    if domain == "cyber":
        # Medir: capacidad de generar exploits, CVEs
        return {"uplift": model.cve_generation_uplift,
                "exploit_uplift": model.exploit_uplift,
                "exfiltration_uplift": model.exfil_uplift}
    if domain == "bio":
        return {"pathogen_uplift": model.pathogen_uplift,
                "synthesis_uplift": model.synthesis_uplift}
    return {}


def threat_model(threat_actor, model_capabilities, defenses):
    """Threat model STRIDE-style."""
    return {
        "spoofing": threat_actor.can_spoof_identity(model_capabilities),
        "tampering": threat_actor.can_modify_outputs(model_capabilities),
        "repudiation": threat_actor.can_deny_actions(model_capabilities),
        "info_disclosure": threat_actor.can_extract(model_capabilities),
        "denial_of_service": threat_actor.can_dos(model_capabilities),
        "elevation": threat_actor.can_escalate(model_capabilities),
    }


def uplift_benchmark(model, baseline, task_suite):
    """Uplift: mejora del atacante con modelo vs sin modelo."""
    with_model = evaluate(model, task_suite)
    without_model = evaluate(baseline, task_suite)
    return with_model / max(without_model, 1e-8)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-dual-use
fase: 18
leccion: 30
---

1. Threat modeling STRIDE-style.
2. Uplift benchmarks por dominio.
3. Controls y mitigations.
4. Red-team en cada release.
```

## Ejercicios

1. **Threat model**: aplicar a tu
   sistema.
2. **Uplift eval**: medir en cyber
   o bio task.
3. **Desafío**: diseñar controls
   end-to-end.

## Lecturas recomendadas

- "Dual Use Foundation Models" (Anthropic 2024)
- "GPT-4 System Card" (OpenAI 2023)
- "Responsible AI for Cyber" (Microsoft)

---

> 📚 **Adaptación al español** de la lección
> "[30-dual-use-risk-cyber-bio-chem-nuclear]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
