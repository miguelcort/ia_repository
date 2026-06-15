# 18 — Frontier safety frameworks: RSP, Pf, FSF

> Frontier safety frameworks son políticas de empresas (Anthropic RSP, OpenAI Preparedness, Meta Frontier Safety) que definen compute thresholds, evals obligatorios, y mitigaciones para riesgos catastróficos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/17, 18/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Conocer RSP, Pf, FSF.
- Implementar compute threshold gates.
- Definir evals obligatorios.
- Diagnosticar compliance.

## Constrúyelo

```python
def rsp_compliance_check(model, eval_thresholds):
    """Anthropic RSP: ASL-2, ASL-3 thresholds."""
    return {
        "capability_threshold": model.capability_estimate
                              >= eval_thresholds["as_l_3"],
        "deployment_safety": model.safety_cases_passed,
        "interpretability_done": model.interpretability_audited,
    }


def compute_threshold_gate(flops, threshold=1e26):
    """Gating training runs por FLOPs compute threshold."""
    if flops >= threshold:
        return {"action": "require_safety_case",
                "evals": ["cyber", "bio", "autonomy"]}
    return {"action": "standard_release"}


def safety_case_report(model, eval_results):
    """Safety case report para frontier release."""
    return {
        "model": model.name,
        "evals_passed": all(r["passed"] for r in eval_results),
        "capabilities": {r["category"]: r["score"]
                        for r in eval_results},
        "residual_risks": [r for r in eval_results
                          if not r["passed"]],
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
name: prompt-rsp
fase: 18
leccion: 18
---

1. Definir compute thresholds.
2. Implementar eval gates.
3. Safety case report.
4. Documentar compliance.
```

## Ejercicios

1. **RSP**: implementar compute gate
   para ASL-2/3.
2. **Safety case**: generar report para
   modelo propio.
3. **Desafío**: diseñar FSF para
   organización.

## Lecturas recomendadas

- "Anthropic Responsible Scaling Policy"
  (Anthropic 2024)
- "OpenAI Preparedness Framework" (OpenAI 2023)
- "Meta Frontier Safety Framework" (Meta 2024)

---

> 📚 **Adaptación al español** de la lección
> "[18-frontier-safety-frameworks-rsp-pf-fsf]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
